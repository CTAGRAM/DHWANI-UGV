#!/usr/bin/env python
"""Run a Newton-trained policy on the rover. numpy only - no CUDA, no TensorRT, no ONNX.

WHY NUMPY AND NOT TensorRT. The obvious deployment route for a Jetson is ONNX -> TensorRT, and
that is what we planned. Then we looked: this Nano's image has **no CUDA and no TensorRT at all**
- no libnvinfer, no /usr/local/cuda, no nvcc, no python bindings. It is a stripped JetRacer
image. Installing the JetPack components would need the rover online and about 2 GB of the 4.7 GB
free.

So we measured the alternative instead. An RL policy is a handful of matrix multiplies, and
numpy 1.13 on the four Cortex-A57 cores runs them at (measured on this robot, 24 Sep 2026):

    48-128-128-6     23 k params    0.140 ms    7137 Hz
    64-256-256-8     84 k params    0.338 ms    2961 Hz
    96-512-512-12   317 k params    1.058 ms     945 Hz

A control loop needs 50-100 Hz. Even the large policy is ten times faster than required. So the
whole CUDA 10.2 problem simply does not arise: train on a GPU, ship the weights, run them here.

THE CONTRACT. Newton (or any trainer) exports a .npz holding, for an N-layer MLP:

    W0, b0, W1, b1, ... W{N-1}, b{N-1}     float32
    obs_mean, obs_std                       optional, for observation normalisation
    act_scale                               optional, output scaling
    activation                              optional, b"tanh" (default) or b"relu"

Nothing else. No framework, no version pinning, no serialisation format that can rot.

SAFETY. This node does not drive by itself. It publishes its action on /policy/action and only
takes control when explicitly enabled, because a policy trained in simulation meeting real
terrain for the first time is exactly the situation where an unsupervised robot does something
expensive.
"""
import math
import os
import numpy as np

# rospy is imported lazily, inside the node. --self-test must run without ROS on the path: it
# validates the numpy inference path, and coupling that check to a ROS install would mean the
# one thing proving deployability cannot run in the environment it is proving.
POLICY = os.environ.get("POLICY", os.path.expanduser("~/models/policy.npz"))
RATE_HZ = 50.0
MAX_SPEED = 0.28          # m/s; the trainer used 0.5, the real base is capped where TEB is capped
MAX_STEER = 0.38          # rad; atan(wheelbase / min turning radius) = atan(0.20 / 0.5)


class Policy(object):
    """An MLP loaded from .npz. Deliberately the least clever thing that works."""

    def __init__(self, path):
        z = np.load(path)
        self.W, self.b = [], []
        i = 0
        while ("W%d" % i) in z:
            self.W.append(z["W%d" % i].astype(np.float32))
            self.b.append(z["b%d" % i].astype(np.float32))
            i += 1
        if not self.W:
            raise ValueError("%s contains no W0/b0 - not a policy in our contract" % path)
        self.mean = z["obs_mean"].astype(np.float32) if "obs_mean" in z else None
        self.std = z["obs_std"].astype(np.float32) if "obs_std" in z else None
        self.scale = float(z["act_scale"]) if "act_scale" in z else 1.0
        act = z["activation"].tobytes().decode() if "activation" in z else "tanh"
        self.act = np.tanh if act.startswith("tanh") else (lambda v: np.maximum(v, 0.0))
        self.obs_dim = self.W[0].shape[0]
        self.act_dim = self.W[-1].shape[1]
        self.n_params = int(sum(w.size for w in self.W) + sum(b.size for b in self.b))

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        if self.mean is not None:
            x = (x - self.mean) / np.maximum(self.std, 1e-6)
        for k in range(len(self.W) - 1):
            x = self.act(x.dot(self.W[k]) + self.b[k])
        return (x.dot(self.W[-1]) + self.b[-1]) * self.scale


def self_test():
    """Prove the path end to end with a random policy, so the runner can be validated before a
    trained one exists. This is what makes 'deployable' a measured claim rather than a plan."""
    import tempfile, time
    print("numpy", np.__version__)
    for obs, h, act in ((48, 128, 6), (64, 256, 8), (96, 512, 12)):
        d = {}
        dims = [obs, h, h, act]
        for k in range(3):
            d["W%d" % k] = np.random.randn(dims[k], dims[k + 1]).astype(np.float32) * 0.1
            d["b%d" % k] = np.zeros(dims[k + 1], np.float32)
        d["obs_mean"] = np.zeros(obs, np.float32)
        d["obs_std"] = np.ones(obs, np.float32)
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez(f.name, **d)
        p = Policy(f.name)
        x = np.random.randn(obs).astype(np.float32)
        for _ in range(50):
            p(x)
        t0 = time.time()
        for _ in range(2000):
            p(x)
        dt = (time.time() - t0) / 2000 * 1000
        print("  %3d-%d-%d-%-3d  %7d params  %.3f ms  %.0f Hz  out%s" %
              (obs, h, h, act, p.n_params, dt, 1000.0 / dt, p(x).shape))
        os.unlink(f.name)
    print("PATH VALIDATED - a trained .npz will run here unchanged.")


class Node(object):
    def __init__(self):
        import rospy
        from std_msgs.msg import Float32MultiArray, String, Bool
        self.rospy = rospy
        self._Arr, self._Str, self._Bool = Float32MultiArray, String, Bool
        from sensor_msgs.msg import Imu
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PointStamped, Twist
        import tf
        self.pub = rospy.Publisher("/policy/action", Float32MultiArray, queue_size=1)
        # Twist the JetRacer base understands: linear.x in m/s, angular.z is the STEERING ANGLE
        # (the base runs with cmd_angle_instead_rotvel). Published on /policy/cmd_vel, not
        # /cmd_vel - wiring it to the wheels is a deliberate remap, never a default.
        self.cmd = rospy.Publisher("/policy/cmd_vel", Twist, queue_size=1)
        self._Twist = Twist
        self.status = rospy.Publisher("/policy/status", String, queue_size=1, latch=True)
        self.enabled = False
        self.imu = self.odom = self.goal = None
        self.prev_a = np.zeros(2, np.float32)
        self.tf = tf.TransformListener()
        rospy.Subscriber("/policy/enable", Bool, self.on_enable, queue_size=1)
        rospy.Subscriber("/imu", Imu, lambda m: setattr(self, "imu", m), queue_size=1)
        rospy.Subscriber("/odom", Odometry, lambda m: setattr(self, "odom", m), queue_size=1)
        rospy.Subscriber("/policy/goal", PointStamped, lambda m: setattr(self, "goal", m), queue_size=1)
        if not os.path.exists(POLICY):
            self.status.publish(self._Str("no policy at %s" % POLICY))
            rospy.logwarn("no policy file at %s - idle", POLICY)
            self.p = None
            return
        self.p = Policy(POLICY)
        rospy.loginfo("policy loaded: %d params, obs %d, act %d",
                      self.p.n_params, self.p.obs_dim, self.p.act_dim)
        self.status.publish(self._Str("loaded %d params" % self.p.n_params))

    def on_enable(self, m):
        # Enabling is explicit and separate from loading. A sim-trained policy meeting real
        # terrain is not something to switch on by merely starting a node.
        self.enabled = bool(m.data)
        self.rospy.logwarn("policy %s", "ENABLED" if self.enabled else "disabled")

    def observe(self):
        """The 14 numbers train/newton_rover_train.py trained on, in its order:
            gravity_body(3) linvel_body(3) angvel_body(3) goal_dir_body(2) goal_dist/5m(1) prev_action(2)
        Every one is something this robot measures today: gravity and rates from /imu, body-frame
        velocity from /odom (twist is already in base_footprint), goal from /policy/goal through tf.
        No wheel encoders, no terrain samples - the trainer deliberately used neither."""
        if self.imu is None or self.odom is None:
            return None
        q = self.imu.orientation
        R = self.tf_quat_to_R(q.x, q.y, q.z, q.w)
        g = R.T.dot(np.array([0.0, 0.0, -1.0], np.float32))
        v = self.odom.twist.twist.linear
        w = self.imu.angular_velocity
        goal_dir, goal_dist = np.zeros(2, np.float32), 0.0
        if self.goal is not None:
            try:
                (tx, ty, _), rot = self.tf.lookupTransform(self.goal.header.frame_id,
                                                           "base_footprint", self.rospy.Time(0))
                dx, dy = self.goal.point.x - tx, self.goal.point.y - ty
                yaw = math.atan2(2.0 * (rot[3] * rot[2] + rot[0] * rot[1]),
                                 1.0 - 2.0 * (rot[1] ** 2 + rot[2] ** 2))
                bx = math.cos(yaw) * dx + math.sin(yaw) * dy
                by = -math.sin(yaw) * dx + math.cos(yaw) * dy
                goal_dist = math.hypot(dx, dy)
                if goal_dist > 1e-3:
                    goal_dir = np.array([bx, by], np.float32) / goal_dist
            except Exception as e:
                self.rospy.logwarn_throttle(5.0, "goal transform failed: %s", e)
        return np.concatenate([g, [v.x, v.y, v.z], [w.x, w.y, w.z], goal_dir,
                               [min(goal_dist, 5.0) / 5.0], self.prev_a]).astype(np.float32)

    @staticmethod
    def tf_quat_to_R(x, y, z, w):
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
                        np.float32)

    def spin(self):
        rospy = self.rospy
        if self.p is None:
            rospy.spin()
            return
        r = rospy.Rate(RATE_HZ)
        while not rospy.is_shutdown():
            if self.enabled:
                obs = self.observe()
                if obs is None:
                    self.rospy.logwarn_throttle(5.0, "no /imu or /odom yet - not acting")
                else:
                    a = np.clip(self.p(obs), -1.0, 1.0)
                    self.prev_a = a.astype(np.float32)
                    m = self._Arr()
                    m.data = [float(v) for v in a]
                    self.pub.publish(m)
                    t = self._Twist()
                    t.linear.x = float(a[0]) * MAX_SPEED
                    t.angular.z = float(a[1]) * MAX_STEER
                    self.cmd.publish(t)
            r.sleep()


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        import rospy
        rospy.init_node("policy_runner")
        Node().spin()

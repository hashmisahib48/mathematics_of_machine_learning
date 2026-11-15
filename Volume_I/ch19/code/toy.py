# Examples 19.3-1 and 19.5-1 by hand: two tests, then a spam filter.
import numpy as np

# Two positive tests, conditionally independent (Eq. 19.3-2)
prior = np.array([0.99, 0.01])           # healthy, sick
p_pos = np.array([0.05, 0.90])           # P(test + | class)
joint = prior * p_pos**2                 # P(class) P(+|class)^2
print("two tests : joint =", joint.round(6),
      " P(sick | ++) =", round(joint[1] / joint.sum(), 4))

# Spam corpus as word counts; columns in alphabetical order
words = ["free", "lunch", "meeting", "now", "prize", "report",
         "win"]
counts = np.array([[1, 0, 0, 1, 1, 0, 0],    # spam messages
                   [1, 0, 0, 0, 1, 0, 1],
                   [1, 0, 0, 2, 0, 0, 1],
                   [0, 0, 1, 0, 0, 1, 0],    # ham messages
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 1, 1, 0, 0, 1, 0]])
y = np.array([1, 1, 1, 0, 0, 0])             # 1 = spam
N = np.array([counts[y == k].sum(0) for k in (0, 1)])   # N_kj

alpha, p = 1.0, len(words)
num = N + alpha                          # numerators of Eq. 19.4-5
den = N.sum(1, keepdims=True) + alpha * p   # 15 ham, 17 spam
theta = num / den
x = np.array([1, 0, 1, 1, 0, 0, 0])      # "free meeting now"
log_post = np.log([0.5, 0.5]) + x @ np.log(theta).T   # Eq. 19.4-6
print("joint (ham, spam) =", np.exp(log_post).round(6))
post = np.exp(log_post - log_post.max()); post /= post.sum()
print("P(spam | message) =", round(post[1], 4))

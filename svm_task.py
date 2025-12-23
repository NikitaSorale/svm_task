# ===============================================
# SVM Task – Parts 1 to 5
# ===============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------------------------
# Part 1: Kernel Selection (RBF Kernel)
# -----------------------------------------------

# Create ring-shaped dataset
X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_scaled, y)

# Plot the data
plt.scatter(X_scaled[y==0,0], X_scaled[y==0,1], color='red', label='Class 0')
plt.scatter(X_scaled[y==1,0], X_scaled[y==1,1], color='blue', label='Class 1')
plt.legend()
plt.title('Part 1: RBF Kernel SVM Data')
plt.show()

# -----------------------------------------------
# Part 2: Hyperparameter Effects (C value)
# -----------------------------------------------

# SVM with C = 1
svm_c1 = SVC(kernel='rbf', C=1.0)
svm_c1.fit(X_scaled, y)

# SVM with C = 10
svm_c10 = SVC(kernel='rbf', C=10.0)
svm_c10.fit(X_scaled, y)

print("Part 2 – Support vectors with C=1:", svm_c1.n_support_)
print("Part 2 – Support vectors with C=10:", svm_c10.n_support_)

# -----------------------------------------------
# Part 3: Support Vector Properties
# -----------------------------------------------

# Show indices of support vectors
print("Part 3 – Indices of support vectors:", svm_rbf.support_)
print("Part 3 – Number of support vectors:", svm_rbf.n_support_)

# Optional: Remove far-away duplicate points
X_new = X_scaled[:150]  # Keep first 150 points (simulate removing duplicates)
y_new = y[:150]

svm_rbf_new = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf_new.fit(X_new, y_new)
print("Part 3 – SV after removing far-away points:", svm_rbf_new.n_support_)

# -----------------------------------------------
# Part 4: Feature Scaling
# -----------------------------------------------

# Make one feature very large to simulate unscaled data
X_unscaled = X.copy()
X_unscaled[:,1] = X_unscaled[:,1] * 1000

# Train SVM without scaling
svm_unscaled = SVC(kernel='rbf', C=1.0)
svm_unscaled.fit(X_unscaled, y)
print("Part 4 – Support vectors without scaling:", svm_unscaled.n_support_)

# Scale features
X_scaled2 = StandardScaler().fit_transform(X_unscaled)

# Train SVM with scaling
svm_scaled = SVC(kernel='rbf', C=1.0)
svm_scaled.fit(X_scaled2, y)
print("Part 4 – Support vectors with scaling:", svm_scaled.n_support_)

# -----------------------------------------------
# Part 5: Decision Function
# -----------------------------------------------

w = np.array([0.5, -1.2])
b = -0.3
x_input = np.array([4, 3])

decision_value = np.dot(w, x_input) + b
print("Part 5 – Decision function value:", decision_value)

if decision_value >= 0:
    print("Part 5 – Predicted class: Positive")
else:
    print("Part 5 – Predicted class: Negative")

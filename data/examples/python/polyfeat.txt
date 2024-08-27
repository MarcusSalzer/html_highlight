from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)

poly = PolynomialFeatures(2)
poly.fit_transform(X)
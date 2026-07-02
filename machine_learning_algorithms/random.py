x = np.random.uniform(0, 10, 100)
noise = np.random.uniform(0, 2, 100)
y = 3*x + noise
df = pd.DataFrame({
    'x': x.flatten(),
    'y': y.flatten()
})
df.head()
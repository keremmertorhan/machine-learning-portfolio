import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#Load dataset
df = pd.read_csv("LifeExpectancy (3).csv")  


life_col = [col for col in df.columns if "Life" in col][0]

#Split into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training dataset contains: {len(train_df)} records")
print(f"Test dataset contains: {len(test_df)} records")

# Histogram and statistics
plt.hist(train_df[life_col].dropna(), bins=20)
plt.title("Life Expectancy Histogram")
plt.xlabel("Life Expectancy")
plt.ylabel("Frequency")
plt.show()

print("\nStatistical Summary of Life Expectancy:")
print(train_df[life_col].describe())

#Top 3 countries by average life expectancy
top_countries = train_df.groupby("Country")[life_col].mean().nlargest(3)
print("\nTop 3 countries with the highest average life expectancy:")
print(top_countries)

#Simple regression models for GDP, Total expenditure, Alcohol
features = ["GDP", "Total expenditure", "Alcohol"]

for feature in features:
    print(f"\n--- Simple Linear Regression for {feature} ---")

    #Clean training data: drop rows where feature or target is NaN
    train_clean = train_df[[feature, life_col]].dropna()
    X_train = train_clean[[feature]]
    y_train = train_clean[life_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    coef = model.coef_[0]
    intercept = model.intercept_
    score = model.score(X_train, y_train)

    print(f"R² Score: {score:.4f}")
    print(f"Equation: y = {coef:.4f}x + {intercept:.4f}")

    plt.scatter(X_train, y_train, alpha=0.3)
    plt.plot(X_train, model.predict(X_train), color='red')
    plt.title(f"{feature} vs Life Expectancy")
    plt.xlabel(feature)
    plt.ylabel("Life Expectancy")
    plt.text(X_train[feature].min(), y_train.max(), f"y = {coef:.2f}x + {intercept:.2f}", color="blue")
    plt.show()

    #Clean test data the same way
    test_clean = test_df[[feature, life_col]].dropna()
    X_test = test_clean[[feature]]
    y_test = test_clean[life_col]

    predictions = model.predict(X_test)
    errors = np.abs(predictions - y_test)

    print(f"Mean Absolute Error: {errors.mean():.2f}")
    print(f"Standard Deviation of Error: {errors.std():.2f}")

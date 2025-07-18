import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体显示和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Low Price'] = pd.to_numeric(df['Low Price'], errors='coerce')
    df['High Price'] = pd.to_numeric(df['High Price'], errors='coerce')
    df['Package Price'] = (df['Low Price'] + df['High Price']) / 2
    df_clean = df[['Package', 'City Name', 'Package Price', 'Variety', 'Date']].dropna()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean['Month'] = df_clean['Date'].dt.month
    return df_clean


def encode_categorical_features(df, categorical_columns):
    encoder = OneHotEncoder(sparse=False)
    encoded_array = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_columns))
    return encoded_df, encoder


def prepare_features(df, encoded_df):
    df_model = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    X = df_model.drop(columns=['Package Price', 'Date', 'Package', 'City Name', 'Variety'])
    y = df_model['Package Price']
    return X, y


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'随机森林回归 RMSE: {rmse:.4f}')
    print(f'随机森林回归 R²: {r2:.4f}')
    return y_pred


def plot_results(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('真实价格')
    plt.ylabel('预测价格')
    plt.title('随机森林预测价格 vs 真实价格')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()


def main():
    filepath = 'US-pumpkins.csv'
    df = load_and_preprocess_data(filepath)
    encoded_df, _ = encode_categorical_features(df, ['Package', 'City Name', 'Variety'])
    X, y = prepare_features(df, encoded_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_random_forest(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)


if __name__ == '__main__':
    main()

import os
import pytest
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
MASTER_MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model_master.pkl")


@pytest.fixture
def prepare_data(test_size=0.2, random_state=42):
    # Titanicデータセットの読み込み
    path = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
    data = pd.read_csv(path)

    # 必要な特徴量の選択と前処理
    data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])  # 性別を数値に変換

    # 整数型の列を浮動小数点型に変換
    data["Pclass"] = data["Pclass"].astype(float)
    data["Sex"] = data["Sex"].astype(float)
    data["Age"] = data["Age"].astype(float)
    data["Fare"] = data["Fare"].astype(float)
    data["Survived"] = data["Survived"].astype(float)

    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def test_compare_models(prepare_data):
    """masterブランチにあるモデルと精度を比較"""
    _, X_test, _, y_test = prepare_data

    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(MASTER_MODEL_PATH)

    with open(MODEL_PATH, "rb") as new_model_f:
        new_model: RandomForestClassifier = pickle.load(new_model_f)
    with open(MASTER_MODEL_PATH, "rb") as old_model_f:
        old_model: RandomForestClassifier = pickle.load(old_model_f)

    print()
    new_predictions = new_model.predict(X_test)
    new_accuracy = accuracy_score(y_test, new_predictions)
    print("new_accuracy", new_accuracy)
    old_predictions = old_model.predict(X_test)
    old_accuracy = accuracy_score(y_test, old_predictions)
    print("old_accuracy", old_accuracy)

    assert new_accuracy >= old_accuracy - 0.2

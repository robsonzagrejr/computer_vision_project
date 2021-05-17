import classic.data as cdata
import classic.model as cmodel

def create_classic_model():
    print("Loading data...")
    df = cdata.load_data()
    print(df.head())
    print("Split data...")
    X_train, X_test, y_train, y_test = cdata.split_train_test(df)
    model = cmodel.train_model(X_train, y_train, search=False)


if __name__ == '__main__':
    create_classic_model()
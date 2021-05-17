import classic as c


def create_classic_model():
    df = c.data.load_data()
    X_train, X_test, y_train, y_test = c.data.split_train_test(df)
    model = c.model.train_model(X_train, y_train, search=False)


if __name__ == '__main__':
    create_classic_model()
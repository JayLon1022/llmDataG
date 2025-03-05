    # 处理缺失值
    train_data = train_data.fillna(train_data.mean())
    test_data = test_data.fillna(test_data.mean())
 # --- Разделение на train и test ---
    # Вариант 1: Просто разделить в пропорции 80/20
    train_size = int(len(prophet_data) * 0.8)
    train = prophet_data[:train_size]
    test = prophet_data[train_size:]

    # Обучаем модель только на train
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(train)

    # Делаем предикт на тестовых датах
    future = test[['ds']]  # Можно использовать и больше, но здесь тестируем на тестовых датах
    forecast = model.predict(future)

    # Объединяем предсказания и реальные значения
    test_forecast = test.merge(forecast[['ds', 'yhat']], on='ds', how='left')

    # Вычисляем метрики
    mae = mean_absolute_error(test_forecast['y'], test_forecast['yhat'])
    rmse = mean_squared_error(test_forecast['y'], test_forecast['yhat'])

    plt.plot(test_forecast["ds"], test_forecast["y"])
    plt.plot(test_forecast["ds"], test_forecast["yhat"])
    plt.show()

    print(product_name)
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
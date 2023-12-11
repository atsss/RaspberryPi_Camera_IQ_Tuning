
Raspberry Pi RasbianOS

# 前提
    RasbianOSが入っている
    WiFiにつながっている※
    IPアドレスが分かっている※
    SSHでログインできる
    SFTPでファイル共有が出来る
    カメラモジュールと通信ができる(libcamera-helloが正常に動く)

    ※この設定と確認のためにシリアルで通信を行っています

# 基本手順（RaspberryPi編）
    RasPiで　gitでファイルをダウンロードする

    git clone https://github.com/nonNoise/RaspberryPi_Camera_Testing.git
    cd RaspberryPi_Camera_Testing/
    python server.py
    
    RaspberryPiのIPアドレス:8080で画像表示サイトが表示される
    画像を左クリックで別名保存ができる
    JSONファイルを書き換えた際は、Ctrl+Cでserver.pyを停止させ、再び実行すると反映される（起動時に反映）

    コンソールに、現在のカメラが設定しているカラーマトリックス値や色温度が表示されている
        ColourTemperature = 3715
        1.02475 , -0.21554 , -0.21446,
        -0.28460 , 1.01909 , -0.21582,
        -0.28539 , -0.21401 , 1.01916


##  １倍の画像を取得する

    python server_1gain.py
    ※imx219_calibration_1gain.jsonを読み込んで起動します
    RaspberryPiのIPアドレス:8080で画像表示サイトが表示される
    画像を左クリックで別名保存ができる
    image.jpg  として保存することが多い

## 作成済み版の画像を取得する

    python server_V2_2207.py
    ※imx219_calibration_V2_2207_mod.jsonを読み込んで起動します
    RaspberryPiのIPアドレス:8080で画像表示サイトが表示される
    画像を左クリックで別名保存ができる
    image_2207.jpg  として保存することが多い


##  Custom版の画像を取得する

    python server.py
    ※imx219_custom.jsonを読み込んで起動します
    RaspberryPiのIPアドレス:8080で画像表示サイトが表示される
    画像を左クリックで別名保存ができる
    image_json.jpg  として保存することが多い
    JSONファイルを書き換えた際は、Ctrl+Cでserver.pyを停止させ、再び実行すると反映される（起動時に反映）


# 基本手順（PC上での操作編）

## CCMの値出力方法

    cd PC_check
    GoProの写真を用意します。
    server_1gainで取得した画像を用意します
    PC_checkフォルダに入れます
    color_checker_V2.py のファイル指定箇所
        IMG_FILE_NAME = 'image.jpg'
        GOPR_FILE_NAME = 'GOPR0070.jpg'
    を変更します。
    
    python color_checker_V2.py

    実行が完了すると

    -------------------------------------------
        1.02475 , -0.21500 , -0.21500 ,
        -0.28500 , 1.01864 , -0.21500 ,
        -0.28500 , -0.21500 , 1.01978

    という表示がされます。この値をJSONに貼り付けます。



## 画像の色差確認方法
    cd PC_check
    GoProの写真を用意します。
    serverで取得した画像を用意します
    PC_checkフォルダに入れます
    color_accuracy_V2.py のファイル指定箇所
        GOPRO_FILE_NAME = 'GOPR0068.jpg'
        IMX_FILE_NAME = 'image_json.jpg'
    を変更します。
    
    python color_accuracy_V2.py

    実行が完了すると

    Total: 136.3164938443946
    Total: 5.679853910183109
    という表示がされます。
    
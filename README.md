> https://medium.com/p/faa179e838fa

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

    git clone git@github.com:atsss/RaspberryPi_Camera_IQ_Tuning.git
    cd RaspberryPi_Camera_IQ_Tuning/raspberrypi/
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

## 作成済み版の画像を取得する

    python server_V2_2207.py
    ※imx219_calibration_V2_2207_mod.jsonを読み込んで起動します
    RaspberryPiのIPアドレス:8080で画像表示サイトが表示される
    画像を左クリックで別名保存ができる


##  Custom版の画像を取得する

    python server.py
    ※imx219_custom.jsonを読み込んで起動します
    RaspberryPiのIPアドレス:8080で画像表示サイトが表示される
    画像を左クリックで別名保存ができる
    JSONファイルを書き換えた際は、Ctrl+Cでserver.pyを停止させ、再び実行すると反映される（起動時に反映）


# 基本手順（PC上での操作編）

## CCMの値出力方法

    cd laptop
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

    cd laptop
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

## JSONの変更方法

    RasPiのフォルダにある　imx219_custom.json　を開きます。

    "rpi.ccm":
            {
                "ccms": [
                    {
                        "ct": 2470,
                        "ccm":
                        [
                            1.02475 , -0.21500 , -0.21500 ,
                            -0.28500 , 1.01864 , -0.21500 ,
                            -0.28500 , -0.21500 , 1.01978
                        ]
                    }
                ]
            }

    ccmの中身をcolor_checker_V2で出力された結果を貼り付けます。
    ctの値は低めにして、確実に反映されるようにします。
    または、server.pyのコンソールに書かれたColourTemperatureの値を入れるのが良いと思います。
    この状態で、JSONを保存し、再度server.pyを実行されると反映される。
    再度画像を保存し、color_accuracy_V2で色差を確認していく作業。

## CCMの変更方法

    color_checker_V2.pyのcreat_ccm_2()にある以下の箇所を変更します

    gain =  0.17
    cnt_offset = 0.88
    ctt_r = 0xff/255.0 *1.0
    ctt_g = 0xd5/255.0 *1.0
    ctt_b = 0xad/255.0 *1.0
    r_offset = 0.155
    g_offset = 0.225
    b_offset = 0.225

    変更後、再度 python color_checker_V2.pyを実行し、出力結果を再度JSONに張ります。

# 環境構築
1. Install pyenv
> https://github.com/pyenv/pyenv
```
brew install pyenv
which pyenv # check
```
```
// bash_profile
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
```

2. Install python 3.11.3
```
pyenv install 3.11.3
```

3. Set v3.11.3 python as local
```
pyenv local 3.11.3
python --version # check
```

4. Install pipenv
```
pip install pipenv
which pipenv # check
```

5. Install libraries
```
pipenv install
pipenv install --dev
```

6. Run
```
pipenv shell
python server.py
```

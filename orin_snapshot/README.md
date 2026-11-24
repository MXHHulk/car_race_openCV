# Orin 設定快照

`lane/config.yaml` 在電腦和 Orin 上不同，Orin 那份才是實際在用的，所以這裡另外存一份。

- `config.yaml`：目前穩定版（`stable-2laps-imu`）的設定，放回 Orin 的 `~/orin_tools/lane/config.yaml`
- `config_5laps.yaml`：`stable-5laps` 的設定（連續五圈，一圈約 92 秒；直線 66、彎道 38、坡上 62、平台 42）
- `config_3laps_82s.yaml`：`stable-3laps-82s` 的設定（連續三圈，一圈 82.2 秒）
- `config_fast_55_36.yaml`：加 IMU 之前的提速版（`dev-fast-55-36`），從 Orin 抓下來的原檔
- `baseline.json`：放回 Orin 的 `~/orin_tools/baseline.json`
- `2026-11-06_level2_2laps/`：整套程式＋設定（第 2 級速度，連續兩圈，一圈 85.6 秒），還原方法見資料夾裡的 README

## 還原穩定版

在 `D:\coding_place\car` 執行，把 `<標籤>` 換成要還原的版本（`git tag -n1` 看全部）：

```bash
git switch --detach <標籤>
scp orin_tools/lane/*.py jetson@<ORIN_IP>:~/orin_tools/lane/
scp orin_snapshot/config.yaml jetson@<ORIN_IP>:~/orin_tools/lane/config.yaml
scp orin_snapshot/baseline.json jetson@<ORIN_IP>:~/orin_tools/baseline.json
git switch main
```

`stable-2laps-imu` 以後的版本需要 IMU（GY-521 接第 17、25、27、28 腳）。IMU 讀不到時程式會照舊跑，但路口、下坡會改回用畫面估車頭角，比較不準。

BMduino 韌體不要用舊標籤重燒：舵機已經改接 PCA9685 CH15，`stable-2laps-v2` 以前的韌體是 CH0。

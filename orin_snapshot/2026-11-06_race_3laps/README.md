# 2026-11-06 比賽起點、連續三圈（run_20261106_184541）

Orin 上實際在跑的整套程式（`~/orin_tools/lane/*.py`、`config.yaml`）和 `baseline.json`，2026-11-06 18:55 從車上抓下來。
從比賽起點（下坡彎道出來的大直線）出發，連續跑完三圈（隧道到隧道 89.4、90.7 秒），之後手動停。用的是滿電電池、舵機連桿鎖緊後。

速度：直線 74、急彎 38、彎道最快 55、前輪打到底 34、坡上 62、坡頂平台 42、岔路段 50、隧道裡 36、出發 3 秒內 50。

和第 3 級（`2026-11-06_level3_2laps`）相比：
- 隧道段 `yaw_offset: -375`（比賽起點大直線 → 隧道直線），第 0 圈 `enter0: 30`（擺車角度有誤差），進過隧道後要到下一圈才再認
- 起步限速 `start_v 50` / `start_s 3`
- 下坡：反打煞車 `down_brake -10`（畫面變化 ≥ `down_brake_m` 才反打、< `down_push_m` 2.5 照 30% 推、中間滑行），
  坡上看鏡頭偏移 `down_off_k 30`，可往左修 `down_lmax 6`，落地右轉 `bend_yaw 45`，保底右轉要離開坡面
- 坡上靠右保護 `ramp_guard 38`
- 新參數（目前都是 1.0，不作用）：`speed_scale`、`steer_scale` / `_r` / `_l`

184541 跑完後才加的兩項（這份快照裡有，還沒實跑驗證）：
- `nodiff_pitch 6`：坡底（俯仰 ≥ 6°）不用差速（進坡停頓）
- `down_brake_m` 5 → 7（坡底前又反打、落地停頓）

## 還原

```bash
scp orin_snapshot/2026-11-06_race_3laps/*.py orin_snapshot/2026-11-06_race_3laps/config.yaml jetson@192.168.0.116:~/orin_tools/lane/
```

```bash
scp orin_snapshot/2026-11-06_race_3laps/baseline.json jetson@192.168.0.116:~/orin_tools/baseline.json
```

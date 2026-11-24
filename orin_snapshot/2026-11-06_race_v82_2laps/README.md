# 穩定版 1：2026-11-06 直線 82、比賽起點連續兩圈（run_20261106_192543）

> 2026-11-07 使用者指定為「穩定版 1」（tag `stable-v1`）。說明見 `路段處理教學.html`。

Orin 上實際在跑的整套程式（`~/orin_tools/lane/*.py`、`config.yaml`）和 `baseline.json`，2026-11-06 19:30 從車上抓下來。
從比賽起點（下坡彎道出來的大直線）出發，連續跑完兩圈（隧道到隧道 87.5 秒），之後手動停。
直線段（≥ 78%）前輪擺動標準差 3.5°，偏移 95% 在 16 cm 內。

速度：直線 82、急彎 40、彎道最快 55、前輪打到底 34、坡上 62、坡頂平台 42、岔路段 50、隧道裡 36、出發 3 秒內 50、進坡前短直線 42。

和 `2026-11-06_race_3laps` 相比：
- 直線 74 → 82，彎道 38 → 40（42 試過，只快 0.8 秒、使用者覺得不穩，退回 40）
- 進坡前短直線限速（`tunnel.py`）：車頭 = 隧道變暗那刻車頭 + `pre_off -200`（±`pre_win 15`），變暗後 8–20 秒內，最多 `pre_v 42`%。
  185702 以 55–65% 進坡前右彎，被甩到左邊 30 cm
- 隧道段進過一次後，同一圈不再認（`lap_need = 目前圈數 + 1`）
- 坡底不用差速 `nodiff_pitch 6`、下坡反打門檻 `down_brake_m 7`（race-3laps 時未實跑，這版已實跑）

## 還原

```bash
scp orin_snapshot/2026-11-06_race_v82_2laps/*.py orin_snapshot/2026-11-06_race_v82_2laps/config.yaml jetson@192.168.0.116:~/orin_tools/lane/
```

```bash
scp orin_snapshot/2026-11-06_race_v82_2laps/baseline.json jetson@192.168.0.116:~/orin_tools/baseline.json
```

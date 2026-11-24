# 2026-11-06 第 2 級速度、連續兩圈（run_20261106_145438）

Orin 上實際在跑的整套程式（`~/orin_tools/lane/*.py`、`config.yaml`）和 `baseline.json`，2026-11-06 15:30 從車上抓下來。
從隧道前的岔路直線出發，連續跑完兩圈（隧道到隧道 85.6 秒），第三圈手動停。

速度（第 2 級）：直線 66、急彎 38、前輪打到底 34、坡上 62、坡頂平台 42、岔路段 50、隧道裡 36。

和 stable-5laps 相比主要多了：
- 隧道段（`tunnel.py`）：畫面變暗認隧道、出發那段就算（`min_lap: 0`）、每過虛線路口用 −112° 重新對準
- 坡上鎖車頭（`ramp_hold_k 0.7`，進坡 1.5 秒後開始，過坡頂放開）、坡上不加前輪置中修正（`climb_trim 0`）
- 單純速度算法也照 `v_sat`（前輪快打到底降到 34%）
- 前輪置中修正 `fwa_trim 6`（舵機中心偏左約 7°）、綠燈起跑（`race_go.py`，`--race`）

## 還原

```bash
scp orin_snapshot/2026-11-06_level2_2laps/*.py orin_snapshot/2026-11-06_level2_2laps/config.yaml jetson@192.168.0.116:~/orin_tools/lane/
```

```bash
scp orin_snapshot/2026-11-06_level2_2laps/baseline.json jetson@192.168.0.116:~/orin_tools/baseline.json
```

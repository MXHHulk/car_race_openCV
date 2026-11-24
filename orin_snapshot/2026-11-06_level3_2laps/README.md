# 2026-11-06 第 3 級速度、連續兩圈（run_20261106_170940）

Orin 上實際在跑的整套程式（`~/orin_tools/lane/*.py`、`config.yaml`）和 `baseline.json`，2026-11-06 17:15 從車上抓下來。
從右彎前的岔路出發（隧道直線的車頭 = 出發方向 −216°），連續跑完兩圈（隧道到隧道 87.4 秒），之後手動停。

速度（第 3 級）：直線 74（第 2 級 66）、急彎 38、彎道最快 55、前輪打到底 34、坡上 62、坡頂平台 42、岔路段 50、隧道裡 36。

和第 2 級（`2026-11-06_level2_2laps`）相比：
- `v_straight` 66 → 74，`v_cmax` 65 → 55（`lat_load` 仍是 0，單純速度算法）
- 隧道段 `yaw_offset: -216`：配合這個起點。**從隧道前的直線出發要改回 0**
- 從岔路直線出發（`min_lap 0`）時，出發頭 1.5 秒內方向對就立刻啟動隧道段（`tunnel.py`，164347 拱門前才啟動、撞右腳）
- 坡上靠右保護 `ramp_xr_after: 1.0`：進坡 1 秒就開始（原本約 3.8 秒；170204 第二圈坡上只剩右線被當左線，3.7 秒壓右護條掉下去）
- `inner_bias.left_max: 25`：右彎中往左最多修 25°（163140 卡在 −15° 開出去）

## 還原

```bash
scp orin_snapshot/2026-11-06_level3_2laps/*.py orin_snapshot/2026-11-06_level3_2laps/config.yaml jetson@192.168.0.116:~/orin_tools/lane/
```

```bash
scp orin_snapshot/2026-11-06_level3_2laps/baseline.json jetson@192.168.0.116:~/orin_tools/baseline.json
```

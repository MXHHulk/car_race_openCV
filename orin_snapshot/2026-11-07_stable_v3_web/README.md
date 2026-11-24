# 穩定版 3 + 網頁（2026-11-07）

行車程式和設定**完全等於穩定版 3**（`stable-v3`：controller / hilltop / tunnel / vision / … 檔案一致，config 只多 `web:` 段）。
另外保留網頁功能：
- `webctl.py`：獨立網頁服務 `python3 webctl.py`，手機開 `http://192.168.0.116:8000/`，可啟動 / 關閉主程式、看即時畫面、開始 / 停止、勾選紀錄與起跑模式
- `webui.py`、`main.py`：主程式只在本機 127.0.0.1:8001 開控制埠；網頁模式每趟按開始到停止、有勾紀錄才寫 CSV + 錄影

之後試過、這版**沒有**的：下坡卡住加推（down_stuck_*）、大直線蛇行偵測（weave_*）。

## 還原

```bash
scp orin_snapshot/2026-11-07_stable_v3_web/*.py orin_snapshot/2026-11-07_stable_v3_web/config.yaml jetson@192.168.0.116:~/orin_tools/lane/
```

# 穩定版 2 + 網頁（2026-11-07）

行車程式和設定**等於穩定版 2**（`stable-v2`：controller / hilltop / tunnel / vision / … 檔案一致，config 只多 `web:` 段），
`main.py` 是穩定版 2 的主程式套上網頁改動（和 `stable-v3-web` 同一份改動）。
另外保留網頁功能：
- `webctl.py`：獨立網頁服務 `python3 webctl.py`，手機開 `http://192.168.0.116:8000/`，可啟動 / 關閉主程式、看即時畫面、開始 / 停止、勾選紀錄與起跑模式
- `webui.py`：主程式只在本機 127.0.0.1:8001 開控制埠；網頁模式每趟按開始到停止、有勾紀錄才寫 CSV + 錄影

速度（穩定版 2）：直線 82、彎道 53、打到底 45；隧道前後、下坡後到大直線彎道 40 / 34（下坡後最多 60）；
隧道 41 / 38 / 27 / 34（進隧道前 / 隧道前直線 / 隧道裡 / 出隧道）。沒有穩定版 3 的進彎前減速。

## 還原

```bash
scp orin_snapshot/2026-11-07_stable_v2_web/*.py orin_snapshot/2026-11-07_stable_v2_web/config.yaml jetson@192.168.0.116:~/orin_tools/lane/
```

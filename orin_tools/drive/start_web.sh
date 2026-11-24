#!/bin/bash
# 開機自動啟動網頁控制台（webctl.py，:8000）。由使用者 crontab 的 @reboot 呼叫，不需要 sudo。
# 手動執行也可以：bash ~/orin_tools/drive/start_web.sh
cd "$(dirname "$0")" || exit 1
LOG=../logs/webctl_boot.log
# 已經在跑就不重複開
if pgrep -f "python3 webctl.py" >/dev/null; then
    echo "$(date '+%F %T') webctl 已經在跑" | tee -a "$LOG"
    exit 0
fi
# 等網路（Wi-Fi 拿到 IP）最多 60 秒，網頁才連得到
for i in $(seq 1 60); do
    if hostname -I 2>/dev/null | grep -q '[0-9]'; then break; fi
    sleep 1
done
echo "$(date '+%F %T') 啟動 webctl（IP：$(hostname -I)），網頁：http://$(hostname -I | awk '{print $1}'):8000" | tee -a "$LOG"
setsid nohup python3 webctl.py >> "$LOG" 2>&1 < /dev/null &

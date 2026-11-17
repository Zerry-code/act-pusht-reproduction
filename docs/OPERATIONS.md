# 从 ZIP 到自己的成果仓库

## 1. 下载与解压

1. 在聊天中点击 `act-pusht-reproduction.zip` 下载。
2. 在 Windows 下载列表或“下载”文件夹中找到 ZIP。
3. 右键 → “全部解压缩” → “提取”。不要在 ZIP 内直接编辑。
4. 解压后应看到一个 `act-pusht-reproduction` 文件夹，里面直接有 README.md、media、scripts 等。
5. 选中这个文件夹，按 Ctrl+C。
6. 在文件管理器地址栏粘贴下面地址并回车：

```text
\\wsl.localhost\Ubuntu-22.04\home\chenzerry\projects
```

7. 按 Ctrl+V，将整个文件夹复制到这里。若已有同名文件夹，先保留旧版，不要直接覆盖。

最终 Ubuntu 路径应是 `~/projects/act-pusht-reproduction/README.md`，避免出现两层同名文件夹。

## 2. 补齐原机证据

打开 PowerShell：

```powershell
wsl -d Ubuntu-22.04
```

在 Ubuntu 中执行：

```bash
cd ~/projects/lerobot
source .venv/bin/activate
bash ~/projects/act-pusht-reproduction/scripts/collect_local.sh
```

这一步不重新训练、不复制大权重、不改动已有实验。它将真实评测 JSON、配置、commit 和环境版本整理进 evidence/。

然后查看文件：

```bash
cd ~/projects/act-pusht-reproduction
cat evidence/collection_manifest.json
explorer.exe "$(wslpath -w "$PWD")"
```

在文件管理器中双击 README.md 可用编辑器阅读。GitHub 上传后会渲染表格和 GIF。

## 3. 建立本地 Git 仓库

确认当前目录是成果仓库，而不是官方 lerobot：

```bash
cd ~/projects/act-pusht-reproduction
pwd
git init -b main
git add .
git status --short
```

`git status` 应包含 README、脚本、media、results、evidence；不应包含模型权重、虚拟环境或数据集。

然后提交：

```bash
git commit -m "Document ACT PushT reproduction and evaluation"
```

如果提示身份未设置，在此仓库内设置真实姓名及 GitHub 隐私邮箱（从 GitHub → Settings → Emails 查看 noreply 地址），再提交。不要照抄占位符，也不用 `--global` 改全部项目。

```bash
git config user.name "你的名字"
git config user.email "你的GitHub提交邮箱"
```

## 4. 上传 GitHub（账号可用后）

1. 浏览器进入 https://github.com/new 并登录。
2. Repository name 填 `act-pusht-reproduction`。
3. Description 可填：`ACT imitation-learning reproduction on PushT with LeRobot, evaluation and demos`。
4. 学习整理阶段可选 Private，准备作为公开作品时选 Public。
5. 不勾选添加 README、.gitignore 或 License，因为本地已有文件。
6. 点击 Create repository。
7. 复制页面上 HTTPS 仓库地址。回到 Ubuntu 终端执行，地址替换为刚复制的真实地址：

```bash
git remote add origin https://github.com/你的用户名/act-pusht-reproduction.git
git push -u origin main
```

这两条需要真实远程地址及认证。仓库创建/推送尚未由助手执行。GitHub 不接受账号密码作为 HTTPS Git 密码；若出现认证阻碍，先反馈提示，不要把 token 发到聊天或写进 URL、README、脚本。

推送后刷新网页，检查 README 表格和两个 GIF。账号无法使用时，先保留 ZIP 和本地仓库，无需为了投递虚构远程链接。

## 5. 本次 WSL 网络问题

Windows 系统代理监听端口为 7897。WSL 内直接访问 127.0.0.1 被拒绝，通过默认网关地址访问已验证成功。仅在当前终端需要联网时设置：

```bash
WIN_HOST_IP=$(ip route show default | awk '{print $3; exit}')
export http_proxy="http://${WIN_HOST_IP}:7897"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
```

保持 Windows 代理软件运行。这是本机已验证方案，不是所有 WSL 网络模式通用配置。无需关闭防火墙；没有把代理配置写入永久启动文件。网络无需修复时不要改设置。

## 6. 修改并验证一个真实参数

阅读 scripts/evaluate.sh。先预测 `n_action_steps` 改变可能怎样影响轨迹，再使用新的输出目录试验，例如：

```bash
bash scripts/evaluate.sh 42 5 10 outputs/eval/portfolio_check_action10
```

执行间隔是运行配置，不是学习到的模型权重。对比时保持权重、种子、回合数等一致。不要降低成功判据来换取更高指标。

## 7. 修复失败 GIF 只有一帧的问题

交付检查发现上传的 seed=42 GIF 只有一帧；原始 seed=42 MP4 在你电脑上，助手没有取得这段原视频，不能凭静态图补造轨迹。复制仓库后，在 Ubuntu 运行：

```bash
REBUILD_GIFS=1 bash ~/projects/act-pusht-reproduction/scripts/make_gifs.sh
```

这会从原始 MP4 重新生成两份 GIF，覆盖仓库内的预览文件，原始 MP4 不变。然后将 GIF 拖到浏览器，检查是否播放；也可运行下面命令检查帧数：

```bash
~/projects/lerobot/.venv/bin/python - <<'PYCODE'
from pathlib import Path
from PIL import Image
for f in (Path.home() / 'projects/act-pusht-reproduction/media').glob('*.gif'):
    with Image.open(f) as image:
        print(f.name, '帧数:', image.n_frames)
PYCODE
```

如果失败文件帧数大于 1 且能播放，将 README 中“失败静态预览”及该文件单帧的说明改为实际状态，再提交。若原始 MP4 自身只有一帧，先反馈文件信息，不要用其它模型的视频冒充此回合。

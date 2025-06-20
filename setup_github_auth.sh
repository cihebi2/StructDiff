#!/bin/bash

# GitHub身份验证设置脚本

echo "=== GitHub身份验证设置 ==="
echo ""
echo "GitHub已停止支持密码验证，需要使用以下方式之一："
echo "1. 个人访问令牌 (Personal Access Token) - 推荐"
echo "2. SSH密钥"
echo ""

# 个人访问令牌设置
setup_token() {
    echo "=== 设置个人访问令牌 ==="
    echo ""
    echo "请按照以下步骤获取GitHub个人访问令牌："
    echo ""
    echo "1. 打开 https://github.com/settings/tokens"
    echo "2. 点击 'Generate new token' -> 'Generate new token (classic)'"
    echo "3. 设置Token名称: StructDiff-$(date +%Y%m%d)"
    echo "4. 选择权限范围 (scopes):"
    echo "   ✓ repo (完整仓库权限)"
    echo "   ✓ workflow (如果需要GitHub Actions)"
    echo "5. 点击 'Generate token'"
    echo "6. 复制生成的令牌 (只显示一次!)"
    echo ""
    echo "生成令牌后，有两种使用方式："
    echo ""
    echo "方式A: 使用 git credential helper (推荐)"
    echo "  git config --global credential.helper store"
    echo "  然后下次push时输入用户名和令牌"
    echo ""
    echo "方式B: 在URL中包含令牌"
    echo "  git remote set-url origin https://YOUR_TOKEN@github.com/cihebi2/StructDiff.git"
    echo ""
    
    read -p "是否要设置credential helper? [y/N]: " setup_helper
    if [[ "$setup_helper" =~ ^[Yy]$ ]]; then
        git config --global credential.helper store
        echo "✓ credential helper已设置"
        echo ""
        echo "现在可以运行推送命令："
        echo "  git push origin main"
        echo ""
        echo "首次推送时会要求输入："
        echo "  Username: 你的GitHub用户名"
        echo "  Password: 刚才获取的个人访问令牌 (不是密码!)"
    fi
}

# SSH密钥设置
setup_ssh() {
    echo "=== 设置SSH密钥 ==="
    echo ""
    
    # 检查是否已有SSH密钥
    if [[ -f ~/.ssh/id_rsa.pub ]]; then
        echo "已存在SSH密钥:"
        echo "$(cat ~/.ssh/id_rsa.pub)"
        echo ""
    else
        echo "生成新的SSH密钥..."
        read -p "输入你的GitHub邮箱: " email
        if [[ -n "$email" ]]; then
            ssh-keygen -t rsa -b 4096 -C "$email" -f ~/.ssh/id_rsa -N ""
            echo "✓ SSH密钥已生成"
        else
            echo "❌ 需要提供邮箱地址"
            return 1
        fi
    fi
    
    echo "请将以下公钥添加到GitHub:"
    echo "1. 打开 https://github.com/settings/keys"
    echo "2. 点击 'New SSH key'"
    echo "3. 复制以下公钥内容:"
    echo ""
    echo "-------- 复制以下内容 --------"
    cat ~/.ssh/id_rsa.pub
    echo "-------- 复制以上内容 --------"
    echo ""
    echo "4. 粘贴到GitHub的Key字段中"
    echo "5. 给密钥起个名字: StructDiff-$(hostname)"
    echo "6. 点击 'Add SSH key'"
    echo ""
    
    read -p "SSH密钥是否已添加到GitHub? [y/N]: " ssh_added
    if [[ "$ssh_added" =~ ^[Yy]$ ]]; then
        # 测试SSH连接
        echo "测试SSH连接..."
        ssh -T git@github.com
        
        # 更改远程URL为SSH
        echo "更改远程仓库URL为SSH..."
        git remote set-url origin git@github.com:cihebi2/StructDiff.git
        echo "✓ 远程URL已更新为SSH"
        
        echo ""
        echo "现在可以运行推送命令："
        echo "  git push origin main"
    fi
}

# 快速推送函数
quick_push() {
    echo "=== 快速推送 ==="
    echo ""
    echo "正在推送代码和标签..."
    
    # 推送主分支
    if git push origin main; then
        echo "✓ 代码推送成功"
        
        # 推送标签
        if git push origin --tags; then
            echo "✓ 标签推送成功"
            
            # 显示最新版本
            latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "无标签")
            echo ""
            echo "🎉 上传完成!"
            echo "最新版本: $latest_tag"
            echo "GitHub: https://github.com/cihebi2/StructDiff"
            if [[ "$latest_tag" != "无标签" ]]; then
                echo "发布页面: https://github.com/cihebi2/StructDiff/releases/tag/$latest_tag"
            fi
        else
            echo "❌ 标签推送失败"
            return 1
        fi
    else
        echo "❌ 代码推送失败"
        echo ""
        echo "可能的原因:"
        echo "1. 未设置身份验证 - 运行: $0 token 或 $0 ssh"
        echo "2. 网络连接问题"
        echo "3. 仓库权限问题"
        return 1
    fi
}

# 主菜单
case ${1:-menu} in
    "token"|"pat")
        setup_token
        ;;
    "ssh")
        setup_ssh
        ;;
    "push")
        quick_push
        ;;
    "menu"|*)
        echo "选择设置方式:"
        echo "1. 个人访问令牌 (推荐) - 输入: $0 token"
        echo "2. SSH密钥 - 输入: $0 ssh"
        echo "3. 直接推送 (如果已设置) - 输入: $0 push"
        echo ""
        read -p "选择 [1/2/3]: " choice
        case $choice in
            1) setup_token ;;
            2) setup_ssh ;;
            3) quick_push ;;
            *) echo "无效选择" ;;
        esac
        ;;
esac 
import os


def update_model(old_name: str, new_name: str):
    if os.path.exists(new_name):
        os.remove(new_name)
    os.rename(old_name, new_name)
    print(f"已将 {old_name} 更新为 {new_name}")


if __name__ == "__main__":
    print("更新模型文件...")
    
    update_model("ecg_best.pth", "ecg_ae.pth")
    update_model("pcg_best.pth", "pcg_ae.pth")
    update_model("cxr_best.pth", "cxr_ae.pth")
    
    print("\n✅ 模型文件更新完成！")

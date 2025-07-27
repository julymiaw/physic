import os


def delete_and_rename(folder, target_size):
    """
    从文件夹开头删除图片，直到剩余大小小于目标大小，并重新编号。
    删除前会输出待删除的文件列表，并询问用户是否确认。

    Args:
        folder: 图片文件夹路径
        target_size: 目标大小 (字节)
    """

    files = sorted(os.listdir(folder))
    files_to_delete = []

    total_size = sum(
        os.path.getsize(os.path.join(folder, file))
        for file in os.listdir(folder)
        if file.endswith(".png")
    )

    delete_size = 0
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(folder, file)
            file_size = os.path.getsize(file_path)
            if total_size - delete_size < target_size:
                break
            else:
                files_to_delete.append(file_path)
                delete_size += file_size

    # 打印待删除文件列表
    print("以下文件将被删除：")
    for file in files_to_delete:
        print(file)

    # 询问用户是否确认
    confirm = input("确认删除吗？(yes/no): ")
    if confirm.lower() == "yes":
        for file in files_to_delete:
            os.remove(file)
        print("文件删除成功！")
    else:
        print("删除操作已取消。")
        return

    # 从第一个剩余文件开始重新编号
    i = 1
    for file in files[files.index(file)]:
        if file.endswith(".png"):
            old_name = os.path.join(folder, file)
            new_name = os.path.join(folder, f"frame_{i:04d}.jpg")
            os.rename(old_name, new_name)
            i += 1


# 设置文件夹路径和目标大小
folder_path = "./images"
target_size = 2 * 1024 * 1024 * 1024

delete_and_rename(folder_path, target_size)

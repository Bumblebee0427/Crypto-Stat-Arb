import os
import re
from pathlib import Path

# 通过当前文件的位置，获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))


# ====================================================================================================
# ** 功能函数 **
# - get_folder_by_root: 获取基于某一个地址的绝对路径
# - get_folder_path: 获取相对于项目根目录的，文件夹的绝对路径
# - get_file_path: 获取相对于项目根目录的，文件的绝对路径
# ====================================================================================================
def get_folder_by_root(root, *paths, auto_create=True) -> str:
    """
    获取基于某一个地址的绝对路径
    :param root: 相对的地址，默认为运行脚本同目录
    :param paths: 路径
    :param auto_create: 是否自动创建需要的文件夹们
    :return: 绝对路径
    """
    _full_path = os.path.join(root, *paths)
    if auto_create and (not os.path.exists(_full_path)):  # 判断文件夹是否存在
        try:
            os.makedirs(_full_path)  # 不存在则创建
        except FileExistsError:
            pass  # 并行过程中，可能造成冲突
    return str(_full_path)


def get_folder_path(*paths, auto_create=True, as_path_type=False) -> str | Path:
    """
    获取相对于项目根目录的，文件夹的绝对路径
    :param paths: 文件夹路径
    :param auto_create: 是否自动创建
    :param as_path_type: 是否返回Path对象
    :return: 文件夹绝对路径
    """
    _p = get_folder_by_root(PROJECT_ROOT, *paths, auto_create=auto_create)
    if as_path_type:
        return Path(_p)
    return _p


def get_file_path(*paths, auto_create=True, as_path_type=False) -> str | Path:
    """
    获取相对于项目根目录的，文件的绝对路径
    :param paths: 文件路径
    :param auto_create: 是否自动创建
    :param as_path_type: 是否返回Path对象
    :return: 文件绝对路径
    """
    parent = get_folder_path(*paths[:-1], auto_create=auto_create, as_path_type=True)
    _p = parent / paths[-1]
    if as_path_type:
        return _p
    return str(_p)


def make_safe_path(path_obj):
    # 将中间的空格替换为下划线
    new_path = Path(re.sub(r'\s+', '_', str(path_obj)))

    # 获取文件名和扩展名
    name = new_path.stem  # 文件名（不包括扩展名）
    suffix = new_path.suffix  # 文件扩展名

    # 仅保留字母、数字和下划线，将文件名转换为安全的名字
    safe_name = re.sub(r'[^\w]', '_', name)

    # 处理 Windows 保留名称
    windows_reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
                              "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2",
                              "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    if safe_name.upper() in windows_reserved_names:
        safe_name = f"{safe_name}_safe"

    # 替换路径中的文件名为安全名字
    safe_path = new_path.with_name(safe_name + suffix)

    return safe_path


if __name__ == '__main__':
    """
    DEMO
    """
    print(get_file_path('data', 'xxx.pkl'))
    print(get_folder_path('logs'))
    print(get_folder_by_root('data', 'center', 'yyds', auto_create=False))

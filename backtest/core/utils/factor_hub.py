import importlib
from pathlib import Path

import pandas as pd


class DummyFactor:
    """
    ！！！！抽象因子对象，仅用于代码提示！！！！
    """
    # 额外数据
    extra_data_dict: dict = {}

    # 因子模式：时序因子/截面因子（factors/sections）
    # False： 时序因子
    # True：  截面因子
    is_cross: bool = False

    def signal(self, *args) -> pd.DataFrame:
        raise NotImplementedError

    def signal_multi_params(self, df, param_list: list | set | tuple) -> dict:
        raise NotImplementedError

    def get_factor_list(self, n):
        raise NotImplementedError


class FactorHub:
    _factor_cache = {}

    # noinspection PyTypeChecker
    @staticmethod
    def get_by_name(factor_name) -> DummyFactor:
        if factor_name in FactorHub._factor_cache:
            return FactorHub._factor_cache[factor_name]

        try:
            try:
                module = 'factors'
                factor_module = importlib.import_module(f"{module}.{factor_name}")
            except ModuleNotFoundError:
                module = 'sections'
                factor_module = importlib.import_module(f"{module}.{factor_name}")

            # 创建一个包含模块变量和函数的字典
            factor_content = {
                name: getattr(factor_module, name) for name in dir(factor_module)
                if not name.startswith("__")
            }
            factor_content['is_cross'] = module == 'sections'

            # 创建一个包含这些变量和函数的对象
            factor_instance = type(factor_name, (), factor_content)

            # 防呆：因子模块为空文件或未实现 signal 时，给出明确报错，避免后续 AttributeError 难定位
            if not hasattr(factor_instance, 'signal'):
                module_file = Path(getattr(factor_module, "__file__", "unknown"))
                file_size = module_file.stat().st_size if module_file.exists() else "unknown"
                raise ValueError(
                    f"Factor `{factor_name}` is missing required method `signal`. "
                    f"module={module}.{factor_name}, file={module_file}, size={file_size}"
                )

            # 缓存策略对象
            FactorHub._factor_cache[factor_name] = factor_instance

            return factor_instance
        except ModuleNotFoundError:
            raise ValueError(f"Factor {factor_name} not found.")
        except AttributeError:
            raise ValueError(f"Error accessing factor content in module {factor_name}.")


# 使用示例
if __name__ == "__main__":
    factor = FactorHub.get_by_name("PctChange")

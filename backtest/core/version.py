from pandas import show_versions

from core.utils.log_kit import logger, divider

sys_version = '1.8.2'
sys_name = 'select-coin'
build_version = f'v{sys_version}.20250930'


def version_prompt():
    show_versions()
    divider('[SYSTEM INFO]', '-', with_timestamp=False)
    logger.debug(f'# VERSION:\t{sys_name}({sys_version})')
    logger.debug(f'# BUILD:\t{build_version}')
    divider('[SYSTEM INFO]', '-', with_timestamp=False)

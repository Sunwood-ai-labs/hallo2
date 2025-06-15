"""
WebUI Tabs Package

各タブコンポーネントのエクスポート
"""

from .animation_tab import create_animation_tab
from .enhancement_tab import create_enhancement_tab
from .usage_tab import create_usage_tab

__all__ = [
    'create_animation_tab',
    'create_enhancement_tab', 
    'create_usage_tab'
]

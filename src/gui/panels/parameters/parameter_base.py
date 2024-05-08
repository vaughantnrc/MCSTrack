import abc
from typing import Final
import wx


class ParameterBaseMetaclass(type(wx.Panel), type(abc.ABC)):
    pass


class ParameterBase(wx.Panel, abc.ABC, metaclass=ParameterBaseMetaclass):

    SPACE_PX_DEFAULT: Final[int] = 10

    label: wx.StaticText | None

    def __init__(
        self,
        parent: wx.Window
    ):
        super().__init__(parent=parent)
        self.label = None

    def add_label_and_space(
        self,
        sizer: wx.Sizer,
        label_text: str,
        space_px: int = SPACE_PX_DEFAULT
    ):
        self.label = wx.StaticText(parent=self, label=label_text)
        sizer.Add(window=self.label, flags=wx.SizerFlags(0).Align(wx.ALIGN_BOTTOM))
        sizer.AddSpacer(size=space_px)

    def disable(self):
        self.set_enabled(enable=False)

    def enable(self):
        self.set_enabled(enable=True)

    @abc.abstractmethod
    def set_enabled(
        self,
        enable: bool
    ):
        if self.label is not None:
            self.label.Enable(enable=enable)

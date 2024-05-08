from typing import Final
import wx


_DEFAULT_TEXTBOX_HEIGHT_PX: Final[int] = 60


class FeedbackTextMultiline(wx.Panel):

    text: wx.TextCtrl

    def __init__(
        self,
        parent: wx.Window,
        label_text: str | None = None,
        height_px: int = _DEFAULT_TEXTBOX_HEIGHT_PX
    ):
        super().__init__(parent=parent)
        main_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)
        if label_text is not None:
            label: wx.StaticText = wx.StaticText(parent=self, label=label_text)
            main_sizer.Add(window=label, flags=wx.SizerFlags(0))
        textbox_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        textbox_vertical_space_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)
        textbox_vertical_space_sizer.AddSpacer(height_px)
        textbox_sizer.Add(sizer=textbox_vertical_space_sizer, flags=wx.SizerFlags(0))
        self.text = wx.TextCtrl(
            parent=self,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH)
        textbox_sizer.Add(window=self.text, flags=wx.SizerFlags(1).Expand())
        main_sizer.Add(sizer=textbox_sizer, flags=wx.SizerFlags(0).Expand())
        clear_button: wx.Button = wx.Button(parent=self, label="Clear")
        main_sizer.Add(window=clear_button, flags=wx.SizerFlags(0).Expand())
        self.SetSizerAndFit(main_sizer)

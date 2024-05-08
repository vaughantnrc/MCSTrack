from src.common.structures import StatusMessage
import wx
import wx.grid


class LogPanel(wx.Panel):
    text: wx.TextCtrl

    def __init__(
        self,
        parent: wx.Window
    ):
        super().__init__(parent=parent)
        sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.text = wx.TextCtrl(
            parent=self,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH)
        self.text.SetEditable(False)
        self.text.SetBackgroundColour(colour=wx.Colour(red=249, green=249, blue=249, alpha=255))
        self.SetSizerAndFit(sizer=sizer)
        sizer.Add(window=self.text, flags=wx.SizerFlags(1).Align(wx.EXPAND))

    def output_status_messages(self, status_messages: list[StatusMessage]):
        for status_message in status_messages:
            line: str = \
                f"[[{status_message.source_label} at " \
                f"{status_message.timestamp_utc_iso8601}]]: " \
                f"{status_message.severity.upper()} - " \
                f"{status_message.message}\n"
            if status_message.severity == "debug":
                self.text.SetDefaultStyle(style=wx.TextAttr(
                    colText=wx.Colour(red=51, green=154, blue=255, alpha=255)))
            elif status_message.severity == "info":
                self.text.SetDefaultStyle(style=wx.TextAttr(
                    colText=wx.BLACK))
            elif status_message.severity == "warning":
                self.text.SetDefaultStyle(style=wx.TextAttr(
                    colText=wx.Colour(red=204, green=154, blue=0, alpha=255)))
            elif status_message.severity == "error":
                self.text.SetDefaultStyle(style=wx.TextAttr(
                    colText=wx.RED))
            elif status_message.severity == "critical":
                self.text.SetDefaultStyle(style=wx.TextAttr(
                    colText=wx.WHITE,
                    colBack=wx.RED))
            self.text.AppendText(line)
            self.text.SetDefaultStyle(style=wx.TextAttr())  # reset

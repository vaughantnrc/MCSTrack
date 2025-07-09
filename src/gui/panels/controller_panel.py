from .base_panel import BasePanel
from .specialized import \
    ConnectionTable, \
    LogPanel
from src.common import \
    DequeueStatusMessagesResponse, \
    StatusMessage, \
    StatusMessageSource
from src.controller import \
    MCTController, \
    ConnectionReport
from typing import Final
import wx
import wx.grid


_STATUS_MESSAGE_TABLE_SUBSCRIBER_LABEL: Final[str] = "status_message_table"


class ControllerPanel(BasePanel):

    _controller: MCTController
    _start_from_configuration_button: wx.Button
    _stop_button: wx.Button
    _connection_table: ConnectionTable
    _controller_status_textbox: wx.TextCtrl
    _log_panel: LogPanel

    _controller_status: str  # last status reported by MCTController
    _connection_reports: list[ConnectionReport]
    _is_updating: bool  # Some things should only trigger during explicit user events

    def __init__(
        self,
        parent: wx.Window,
        controller: MCTController,
        status_message_source: StatusMessageSource,
        name: str = "ControllerPanel"
    ):
        super().__init__(
            parent=parent,
            status_message_source=status_message_source,
            name=name)
        self._controller = controller

        self._controller.add_status_subscriber(client_identifier=_STATUS_MESSAGE_TABLE_SUBSCRIBER_LABEL)
        self.status_message_source.add_status_subscriber(subscriber_label=_STATUS_MESSAGE_TABLE_SUBSCRIBER_LABEL)

        horizontal_split_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        control_border_panel: wx.Panel = wx.Panel(parent=self)
        control_border_box: wx.StaticBoxSizer = wx.StaticBoxSizer(
            orient=wx.VERTICAL,
            parent=control_border_panel)
        control_panel: wx.ScrolledWindow = wx.ScrolledWindow(
            parent=control_border_panel)
        control_panel.SetScrollRate(
            xstep=1,
            ystep=1)
        control_panel.ShowScrollbars(
            horz=wx.SHOW_SB_NEVER,
            vert=wx.SHOW_SB_ALWAYS)

        control_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self._start_from_configuration_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Start From File")

        self._stop_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Stop")

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self._controller_status_textbox = wx.TextCtrl(
            parent=control_panel,
            style=wx.TE_READONLY | wx.TE_RICH)
        self._controller_status_textbox.SetEditable(False)
        self._controller_status_textbox.SetBackgroundColour(colour=wx.Colour(red=249, green=249, blue=249, alpha=255))
        control_sizer.Add(
            window=self._controller_status_textbox,
            flags=wx.SizerFlags(0).Expand())
        control_sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)

        self._connection_table = ConnectionTable(parent=control_panel)
        control_sizer.Add(
            window=self._connection_table,
            flags=wx.SizerFlags(0).Expand())
        control_sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)

        control_spacer_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        control_sizer.Add(
            sizer=control_spacer_sizer,
            flags=wx.SizerFlags(1).Expand())

        control_panel.SetSizerAndFit(sizer=control_sizer)
        control_border_box.Add(
            window=control_panel,
            flags=wx.SizerFlags(1).Expand())
        control_border_panel.SetSizer(sizer=control_border_box)
        horizontal_split_sizer.Add(
            window=control_border_panel,
            flags=wx.SizerFlags(35).Expand())

        self._log_panel = LogPanel(parent=self)
        self._log_panel.SetBackgroundColour(colour=wx.BLACK)
        horizontal_split_sizer.Add(
            window=self._log_panel,
            flags=wx.SizerFlags(65).Expand())

        self.SetSizerAndFit(sizer=horizontal_split_sizer)

        self._start_from_configuration_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_start_from_configuration_pressed)
        self._stop_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_stop_pressed)

        self._controller_status = str()
        self._connection_reports = list()
        self._is_updating = False

        self.update_controller_buttons()

    def on_start_from_configuration_pressed(self, _event: wx.CommandEvent) -> None:
        dialog: wx.FileDialog = wx.FileDialog(
            parent=self,
            message="Select a configuration file",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        self._controller.start_from_configuration_filepath(dialog.GetPath())
        self.update_controller_buttons()

    def on_stop_pressed(self, _event: wx.CommandEvent) -> None:
        self._controller.shut_down()
        self.update_controller_buttons()

    def on_remove_pressed(self, _event: wx.CommandEvent):
        selected_row_label: str | None = self._connection_table.get_selected_row_label()
        self._controller.remove_connection(label=selected_row_label)
        self.update_connection_table_display()
        self.update_controller_buttons()

    def update_loop(self):
        super().update_loop()
        self._is_updating = True
        self.update_connection_table_display()
        controller_status: str = self._controller.get_status()
        if controller_status != self._controller_status:
            self._controller_status = controller_status
            self._controller_status_textbox.SetValue(f"MCTController Status: {controller_status}")
            self.update_controller_buttons()
        self.update_loop_log_table()
        self._is_updating = False

    def update_controller_buttons(self):
        self._start_from_configuration_button.Enable(enable=False)
        self._stop_button.Enable(enable=False)
        if self._controller.is_running():
            self._stop_button.Enable(enable=True)
        elif self._controller.is_idle():
            self._start_from_configuration_button.Enable(enable=True)

    def update_connection_table_display(self) -> None:
        # Return if there is no change
        connection_reports: list[ConnectionReport] = self._controller.get_connection_reports()
        if len(connection_reports) == len(self._connection_reports):
            identical: bool = True
            for connection_report in connection_reports:
                contained: bool = connection_report in self._connection_reports
                if not contained:
                    identical = False
                    break
            if identical:
                return
        # There has been a change so update internal variables and UI
        self._connection_reports = connection_reports
        self._connection_table.update_contents(row_contents=self._connection_reports)
        selected_row_index: int | None = self._connection_table.get_selected_row_index()
        if selected_row_index is not None and selected_row_index >= len(self._connection_reports):
            selected_row_index = None
        self._connection_table.set_selected_row_index(selected_row_index)

    def update_loop_log_table(self):
        status_messages_response: DequeueStatusMessagesResponse = \
            self._controller.dequeue_status_messages(
                client_identifier=_STATUS_MESSAGE_TABLE_SUBSCRIBER_LABEL)
        status_messages: list[StatusMessage] = status_messages_response.status_messages
        status_messages += self.status_message_source.pop_new_status_messages(
            subscriber_label=_STATUS_MESSAGE_TABLE_SUBSCRIBER_LABEL)
        self._log_panel.output_status_messages(status_messages=status_messages)

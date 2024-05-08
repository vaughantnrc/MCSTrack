import abc
from typing import Final, Generic, TypeVar
import wx
import wx.grid


T = TypeVar("T")


class _RowSelectionTableMeta(type(wx.Panel), abc.ABCMeta):
    pass  # addresses TypeError: metaclass conflict


class RowSelectionTable(wx.Panel, Generic[T], metaclass=_RowSelectionTableMeta):
    table: wx.grid.Grid
    col_labels: list[str]
    selected_index: int | None

    DEFAULT_HEIGHT_PX: Final[int] = 128

    def __init__(
        self,
        parent: wx.Window,
        col_labels: list[str] | None = None,
        height_px: int = DEFAULT_HEIGHT_PX
    ):
        super().__init__(parent=parent)
        sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        fixed_height: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)
        fixed_height.AddSpacer(size=height_px)
        sizer.Add(sizer=fixed_height)
        self.table = wx.grid.Grid(parent=self)
        if col_labels is None:
            self.col_labels = list()
            self.table.HideColLabels()
        else:
            self.col_labels = col_labels
            self.table.CreateGrid(numRows=0, numCols=len(self.col_labels))
            for col_index in range(0, len(col_labels)):
                self.table.SetColLabelValue(col=col_index, value=col_labels[col_index])
        self.table.HideRowLabels()
        self.table.SetSelectionMode(selmode=wx.grid.Grid.SelectRows)
        self.table.EnableEditing(edit=False)
        self.SetSizer(sizer=sizer)
        sizer.Add(window=self.table, flags=wx.SizerFlags(1).Align(wx.EXPAND))
        self.auto_size_columns_and_fit()
        self.selected_index = None
        self.table.Bind(
            event=wx.grid.EVT_GRID_SELECT_CELL,
            handler=self._on_row_selected)

    def auto_size_columns_and_fit(self):
        for col_index in range(0, len(self.col_labels)):
            self.table.AutoSizeColumn(col=col_index, setAsMin=True)
        self.table.GetParent().Fit()

    def get_selected_row_index(self) -> int | None:
        row_indices: list[int] = self.table.GetSelectedRows()
        if len(row_indices) <= 0:
            return None
        return row_indices[0]

    def get_selected_row_label(self) -> str | None:
        row_index = self.get_selected_row_index()
        if row_index is None:
            return None
        return self.table.GetCellValue(row=row_index, col=0)

    # If idx not specified, then deselects
    def set_selected_row_index(self, idx: int | None = None) -> None:
        if idx is None or idx < 0 or idx >= self.table.GetNumberRows():
            self.selected_index = None
        else:
            self.selected_index = idx
        self.update_selection()

    def update_contents(
        self,
        row_contents: list[T]
    ):
        self.table.ClearSelection()
        row_count: int = self.table.GetNumberRows()
        if row_count > 0:
            self.table.DeleteRows(numRows=row_count)

        self.table.AppendRows(numRows=len(row_contents))
        for row_index, row_content in enumerate(row_contents):
            self._set_row_contents(row_index, row_content)
        self.auto_size_columns_and_fit()
        self.update_selection()

    def update_selection(self) -> None:
        self.table.ClearSelection()
        if self.selected_index is not None and 0 <= self.selected_index < self.table.GetNumberRows():
            self.table.SelectRow(row=self.selected_index)

    def _on_row_selected(self, event: wx.grid.GridEvent) -> None:
        row_index: int = event.GetRow()
        if event.Selecting():
            self.set_selected_row_index(row_index)
        else:
            self.set_selected_row_index(None)

    @abc.abstractmethod
    def _set_row_contents(
        self,
        row_index: int,
        row_content: T
    ):
        pass

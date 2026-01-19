import win32com.client as win32
import pandas as pd
from typing import List, Optional, Union, Tuple, Iterable, Any, Dict
import os
from pandas.io.formats.style import Styler
import io
import logging
import traceback
import threading
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import tempfile


def fig_to_png_bytes(fig, scale: int = 2) -> bytes:
    # Requires kaleido: pip install -U kaleido
    return fig.to_image(format="png", scale=scale)


def fig_to_png_bytes_fast(fig,
                          *,
                          width: int | None = None,
                          height: int | None = None,
                          scale: int = 1,
                          validate: bool = False) -> bytes:
    """
    Faster Plotly → PNG:
    - keeps MathJax off
    - avoids oversized 'scale'
    - skips schema validation
    """
    # Keep images modest unless you truly need retina sizes
    return fig.to_image(
        format="png",
        width=width,
        height=height,
        scale=scale,        # avoid 2–3 unless necessary
        engine="kaleido",
        validate=validate   # skip validation for speed
    )


class OutlookEmailSender:
    def __init__(self):
        self.outlook = win32.Dispatch('outlook.application')


    def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        html_body: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        attachments: Optional[List[str]] = None,
        display_before_send: bool = False,
        *,
        # NEW: inline (CID) attachments
        cid_attachments: Optional[Iterable[Dict[str, Any]]] = None,
    ):
        """
        cid_attachments: iterable of dicts with keys:
          - cid (str): content id used in <img src="cid:...">
          - source (bytes | str): PNG/JPEG bytes or a file path
          - subtype (str): 'png' | 'jpeg' | etc. (only used to pick filename when source is bytes)
          - filename (optional str): nice name for the attachment (Outlook requires a file path; we may temp-save)
        """
        mail = self.outlook.CreateItem(0)  # 0 = olMailItem
        mail.To = self._format_recipients(to)
        mail.Subject = subject
        mail.HTMLBody = html_body

        if cc:
            mail.CC = self._format_recipients(cc)
        if bcc:
            mail.BCC = self._format_recipients(bcc)

        # Regular file attachments
        if attachments:
            for file_path in attachments:
                if file_path and os.path.isfile(file_path):
                    mail.Attachments.Add(file_path)
                else:
                    print(f"[Warning] Attachment not found or invalid: {file_path}")

        # Inline CID attachments
        tmp_files = []  # keep references so they aren't GC'd before send
        if cid_attachments:
            for item in cid_attachments:
                cid = item["cid"]
                source = item["source"]
                subtype = item.get("subtype", "png")
                filename = item.get("filename", f"{cid}.{subtype}")
                path = self._ensure_file_path(source, suffix=f".{subtype}", filename=filename, tmp_files=tmp_files)
                self._attach_with_cid(mail, path, cid)

        if display_before_send:
            mail.Display()
        else:
            mail.Send()

        # best-effort cleanup of temp files
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass

    # ---------- NEW HELPERS ----------

    @staticmethod
    def _ensure_file_path(source: Union[bytes, str], *, suffix: str, filename: str, tmp_files: list) -> str:
        """
        Outlook COM Attachments.Add needs a filesystem path.
        If source is bytes, write to a temp file and return its path.
        If source is a string, assume it's an existing path and return it.
        """
        if isinstance(source, (bytes, bytearray)):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(source)
            tmp.flush()
            tmp.close()
            tmp_files.append(tmp.name)
            return tmp.name
        elif isinstance(source, str):
            # user supplied a file path
            if not os.path.isfile(source):
                raise FileNotFoundError(f"CID attachment path not found: {source}")
            return source
        else:
            raise TypeError("cid_attachments 'source' must be bytes or a file path string.")

    @staticmethod
    def _attach_with_cid(mail, file_path: str, cid: str) -> None:
        """
        Attach a file and set the MAPI Content-ID so <img src="cid:..."> resolves inline.
        """
        attachment = mail.Attachments.Add(Source=file_path)
        PA = attachment.PropertyAccessor
        # PR_ATTACH_CONTENT_ID (0x3712) type PT_STRING8 => ...001E
        PR_ATTACH_CONTENT_ID = "http://schemas.microsoft.com/mapi/proptag/0x3712001E"
        PA.SetProperty(PR_ATTACH_CONTENT_ID, cid)

    # ---------- existing helpers (unchanged) ----------

    @staticmethod
    def _default_styler(
        df: pd.DataFrame,
        default_format: str = "{:,.2f}",
        column_format_dict: Optional[dict] = None
    ) -> Styler:
        # (your original implementation)
        ...

    @staticmethod
    def _format_recipients(recipients: Union[str, List[str]]) -> str:
        if isinstance(recipients, list):
            return '; '.join(recipients)
        return recipients

    @staticmethod
    def parentheses_format(x, precision=2, prefix=''):
        if pd.isna(x):
            return ""
        fmt = f"{prefix}{{:,.{precision}f}}"
        return f"({fmt.format(-x)})" if x < 0 else fmt.format(x)

    def dataframe_to_html(
            self,
            df: pd.DataFrame,
            float_format: str = "{:,.2f}",
            column_format_dict: Optional[dict] = None,
            style: bool = True
    ) -> str:
        """Convert DataFrame to styled HTML table for embedding in email."""
        if style:
            styled_df = self._default_styler(df, default_format=float_format, column_format_dict=column_format_dict)
            table_html = styled_df.to_html()
        else:
            table_html = df.to_html(index=False, border=0)
        return table_html

    @staticmethod
    def _default_styler(
            df: pd.DataFrame,
            default_format: str = "{:,.2f}",
            column_format_dict: Optional[dict] = None
    ) -> Styler:
        """Apply styling to pandas DataFrame with optional per-column formats."""
        # Set up formatting
        formatter = {}
        for col in df.columns:
            if column_format_dict and col in column_format_dict:
                formatter[col] = column_format_dict[col]
            elif pd.api.types.is_numeric_dtype(df[col]):
                formatter[col] = default_format

        def red_negative(val):
            if isinstance(val, (int, float)) and val < 0:
                return "color: red;"
            return ""

        # Build the styled DataFrame
        styled = df.style \
            .format(formatter) \
            .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ]) \
            .set_properties(**{'border': '1px solid lightgray', 'padding': '6px'}) \
            .hide_index() \
            .applymap_index(lambda _: 'font-weight: bold;', axis=1) \
            .applymap_index(lambda _: 'font-weight: bold;', axis=0)

        # Apply red font for negative values (only to numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns
        styled = styled.applymap(red_negative, subset=numeric_cols)

        return styled

    @staticmethod
    def _format_recipients(recipients: Union[str, List[str]]) -> str:
        if isinstance(recipients, list):
            return '; '.join(recipients)
        return recipients

    @staticmethod
    def parentheses_format(x, precision=2, prefix=''):
        """Format numbers with parentheses for negatives, optional prefix like $."""
        if pd.isna(x):
            return ""
        fmt = f"{prefix}{{:,.{precision}f}}"
        return f"({fmt.format(-x)})" if x < 0 else fmt.format(x)



# --- small helper: capture logs from this thread only ---
class _ThreadFilter(logging.Filter):
    def __init__(self, thread_id: int):
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        # Only keep records emitted by this thread
        return record.thread == self.thread_id


@contextmanager
def _capture_logs(level=logging.INFO, logger: logging.Logger | None = None):
    """
    Temporarily attach a StreamHandler to capture logs emitted during the context.
    Captures root logger by default. Filters by current thread to avoid cross-talk.
    """
    log = logger or logging.getLogger()  # root by default
    stream = io.StringIO()

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.addFilter(_ThreadFilter(threading.get_ident()))
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Preserve original level to restore later (root may be WARNING by default)
    original_level = log.level
    try:
        if original_level > level:
            log.setLevel(level)
        log.addHandler(handler)
        yield stream
    finally:
        # Always detach and restore
        log.removeHandler(handler)
        log.setLevel(original_level)
        handler.flush()


def notify_on_failure(
    to,
    *,
    capture_level=logging.INFO,
    include_logs=True,
    max_log_chars=40_000,
    logger: logging.Logger | None = None,
):
    """
    Decorator: run func, email success/failure, and (optionally) include captured logs.

    Parameters
    ----------
    to : str | list[str]
        Email recipient(s).
    capture_level : logging level
        Minimum log level to capture (e.g., logging.INFO or logging.DEBUG).
    include_logs : bool
        Whether to include logs in the email body.
    max_log_chars : int
        Truncate captured logs to this many characters (tail is kept).
    logger : logging.Logger | None
        Which logger to attach to (defaults to root).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Capture logs for just this call
            cm = _capture_logs(level=capture_level, logger=logger) if include_logs else contextmanager(lambda: (yield None))()
            with cm as log_stream:
                try:
                    result = func(*args, **kwargs)

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    captured = log_stream.getvalue() if (include_logs and log_stream) else ""
                    if captured and len(captured) > max_log_chars:
                        captured = f"...[truncated]\n{captured[-max_log_chars:]}"

                    logs_html = f"""
                    <details style="margin-top:10px;">
                      <summary><b>Execution Logs (min level: {logging.getLevelName(capture_level)})</b></summary>
                      <pre style="white-space:pre-wrap;background:#fafafa;border:1px solid #eee;padding:10px;">{captured or "No logs captured."}</pre>
                    </details>
                    """ if include_logs else ""

                    html_body = f"""
                    <h3>Job Succeeded</h3>
                    <p><b>Function:</b> {func.__name__}</p>
                    <p><b>Time:</b> {timestamp}</p>
                    <p>The function executed successfully.</p>
                    {logs_html}
                    """

                    OutlookEmailSender().send_email(
                        to=to,
                        subject=f"Job Success: {func.__name__}",
                        html_body=html_body
                    )
                    return result

                except Exception as e:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    error_message = traceback.format_exc()

                    captured = log_stream.getvalue() if (include_logs and log_stream) else ""
                    if captured and len(captured) > max_log_chars:
                        captured = f"...[truncated]\n{captured[-max_log_chars:]}"

                    logs_html = f"""
                    <details open style="margin-top:10px;">
                      <summary><b>Execution Logs (min level: {logging.getLevelName(capture_level)})</b></summary>
                      <pre style="white-space:pre-wrap;background:#fff6f6;border:1px solid #f3dcdc;padding:10px;">{captured or "No logs captured."}</pre>
                    </details>
                    """ if include_logs else ""

                    html_body = f"""
                    <h3>Job Failed</h3>
                    <p><b>Function:</b> {func.__name__}</p>
                    <p><b>Time:</b> {timestamp}</p>
                    <p><b>Error:</b> {e}</p>
                    <pre style="white-space:pre-wrap;color:#b00020;background:#fff6f6;border:1px solid #f3dcdc;padding:10px;">{error_message}</pre>
                    {logs_html}
                    """

                    OutlookEmailSender().send_email(
                        to=to,
                        subject=f"Job Failed: {func.__name__}",
                        html_body=html_body
                    )
                    raise  # re-raise after notifying
        return wrapper
    return decorator


from html import escape
import numpy as np


def _lerp(a, b, t):
    return a + (b - a) * t

def _rgb(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def _cmap_diverging(t: float) -> str:
    """
    0 -> red, 0.5 -> yellow, 1 -> green
    Simple, email-safe color ramp.
    """
    t = min(max(t, 0.0), 1.0)
    if t <= 0.5:  # red -> yellow
        u = t / 0.5
        r = 255
        g = _lerp(0, 255, u)
        b = 0
    else:  # yellow -> green
        u = (t - 0.5) / 0.5
        r = _lerp(255, 0, u)
        g = 255
        b = 0
    return _rgb(r, g, b)

def df_to_inline_html(
    df: pd.DataFrame,
    column_cfg: dict | None = None,
    decimals: int = 2,
    color_cols: list[str] | None = None,   # NEW: whitelist
    skip_cols: list[str] | None = None,    # NEW: blacklist
) -> str:
    column_cfg = column_cfg or {}
    skip_cols = set(skip_cols or [])
    whitelist = set(color_cols) if color_cols is not None else None

    # numeric columns eligible for coloring
    num_cols_all = df.select_dtypes(include="number").columns
    num_cols = [
        c for c in num_cols_all
        if (whitelist is None or c in whitelist)
        and c not in skip_cols
        and not column_cfg.get(c, {}).get("skip", False)
    ]

    # --- the rest is unchanged, but make sure you:
    # 1) build `scales` only for `num_cols`
    # 2) apply bg only if `c in num_cols`
    scales = {}
    for c in num_cols:
        cfg = column_cfg.get(c, {})
        vmin = cfg.get("vmin", np.nanmin(df[c].values.astype(float)))
        vmax = cfg.get("vmax", np.nanmax(df[c].values.astype(float)))
        center = cfg.get("center", None)
        reverse = cfg.get("reverse", False)
        if center is not None:
            span = max(abs(vmax - center), abs(vmin - center))
            vmin, vmax = center - span, center + span
        if vmax == vmin:
            vmax = vmin + 1e-12
        scales[c] = (float(vmin), float(vmax), bool(reverse))

    ths = "".join(f"<th style='border:1px solid #ddd; padding:6px; text-align:center; background:#fafafa;'>{escape(str(col))}</th>"
                  for col in df.columns)

    rows = []
    for _, r in df.iterrows():
        tds = []
        for c, v in r.items():
            text = "" if pd.isna(v) else (f"{v:.{decimals}f}" if isinstance(v, (int,float,np.number)) else escape(str(v)))
            style = "border:1px solid #ddd; padding:6px; white-space:nowrap;"
            if c in num_cols and not pd.isna(v):
                vmin, vmax, reverse = scales[c]
                t = (float(v) - vmin) / (vmax - vmin)
                if reverse: t = 1 - t
                style += f" background-color:{_cmap_diverging(t)};"
            elif pd.isna(v):
                style += " background-color:#f2f2f2;"
            tds.append(f"<td style='{style}'>{text}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")

    return (
        "<table cellpadding='0' cellspacing='0' "
        "style='border-collapse:collapse; font-family:Arial, Helvetica, sans-serif; font-size:12px;'>"
        f"<thead><tr>{ths}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
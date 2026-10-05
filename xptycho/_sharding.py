"""How a reconstruction is divided among devices, and the two operations
that move data between them.  See the Computing page of the design.

Three terms:

block
    The positions are sorted by the first row of their patch and cut into
    one block per device, of nearly equal count.
band
    The rows of the object image are cut into one band per device, of
    nearly equal height.  A device owns its band.
window
    The range of image rows that the patches of a device's block touch.
"""
import numpy as np
import torch


class Layout:
    """The blocks, bands, and windows of one run.

    Args:
        devices (list of torch.device): one entry per shard.  The same
            device may appear more than once, which is how the tests run
            several shards on one CPU.
        starts (ndarray): int ``(J, 2)``, the row and column of the first
            pixel of each patch on the object grid.
        frame_size (int): the patches are ``frame_size`` square.
        object_shape (tuple of int): ``(rows, cols)`` of the object grid.

    Attributes:
        blocks (list of ndarray): for each device, the indices of its
            positions in the scan's own order.
        bands (list of tuple): for each device, the ``(first, stop)`` rows
            it owns.
        windows (list of tuple): for each device, the ``(first, stop)``
            rows its patches touch.
    """

    def __init__(self, devices, starts, frame_size, object_shape):
        self.devices = [torch.device(d) for d in devices]
        n = len(self.devices)
        rows = int(object_shape[0])
        if n > len(starts):
            raise ValueError('{} devices were asked for but the scan has only {} positions'.format(n, len(starts)))
        if n > rows:
            raise ValueError('{} devices were asked for but the object has only {} rows'.format(n, rows))
        order = np.argsort(starts[:, 0], kind='stable')
        self.blocks = np.array_split(order, n)
        edges = np.linspace(0, rows, n + 1).round().astype(int)
        self.bands = [(int(edges[g]), int(edges[g + 1])) for g in range(n)]
        self.windows = [(int(starts[b, 0].min()), int(starts[b, 0].max()) + frame_size) for b in self.blocks]
        self.object_shape = (rows, int(object_shape[1]))

    def __len__(self):
        return len(self.devices)

    def _overlap(self, window, band):
        first, stop = max(window[0], band[0]), min(window[1], band[1])
        return (first, stop) if stop > first else None

    def sum_to_owner(self, parts):
        """Add the devices' partial sums onto the owners of the rows.

        Args:
            parts (list of Tensor): for each device, its partial sum over
                its window, shape ``(..., window rows, cols)``.

        Returns:
            list of Tensor: for each device, the complete sum over its
            band, shape ``(..., band rows, cols)``.  The pieces are added
            in device order.
        """
        out = []
        for g, band in enumerate(self.bands):
            lead = parts[g].shape[:-2]
            total = torch.zeros(lead + (band[1] - band[0], self.object_shape[1]),
                                dtype=parts[g].dtype, device=self.devices[g])
            for h, window in enumerate(self.windows):
                rows = self._overlap(window, band)
                if rows is not None:
                    piece = parts[h][..., rows[0] - window[0]:rows[1] - window[0], :]
                    total[..., rows[0] - band[0]:rows[1] - band[0], :] += piece.to(self.devices[g])
            out.append(total)
        return out

    def copy_to_window(self, bands):
        """Give each device the rows of its window, from their owners.

        Args:
            bands (list of Tensor): for each device, an array over its band.

        Returns:
            list of Tensor: for each device, the same array over its window.
        """
        out = []
        for g, window in enumerate(self.windows):
            pieces = []
            for h, band in enumerate(self.bands):
                rows = self._overlap(window, band)
                if rows is not None:
                    pieces.append(bands[h][..., rows[0] - band[0]:rows[1] - band[0], :].to(self.devices[g]))
            out.append(torch.cat(pieces, dim=-2))
        return out

    def sum_small(self, parts):
        """Add small partial sums on the first device, in device order."""
        total = parts[0].clone()
        for part in parts[1:]:
            total += part.to(self.devices[0])
        return total

    def broadcast(self, x):
        """A copy of a small array on every device."""
        return [x.to(d) for d in self.devices]

    def gather(self, bands):
        """Join the bands into one image on the host, as numpy."""
        return np.concatenate([b.cpu().numpy() for b in bands], axis=-2)

    def split(self, image):
        """Cut a host image into bands and place each on its owner."""
        image = torch.as_tensor(image)
        return [image[..., first:stop, :].to(d) for (first, stop), d in zip(self.bands, self.devices)]

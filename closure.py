from collections import deque


class Closure:
    """Maintain the materialized nonempty-path closure C(x,z) <- E(x,z)
    or C(x,y) and E(y,z).

    reach(x,z).support counts the distinct final edges E(y,z) that witness a
    nonempty path x -> ... -> y -> z in the current edge set.  A relation is
    present exactly when its support is positive, so self reachability holds
    only when a cycle exists.

    Both insert and delete are local DRed over the cone the changed edge can
    witness: the closure facts the edge may affect are located through the
    existing closure and its forward/reverse indexes, over-deleted, then
    re-derived from the surviving closure with an exact distinct-witness
    support.  Relations with an alternative path are restored; a relation whose
    last path disappears stays deleted.  Maintenance never rebuilds the table,
    replays history, walks a source's stable branches, or scans unrelated parts.
    """

    def __init__(self, store):
        self.store = store

    def _rederive(self, affected):
        """Recompute exact support for every over-deleted fact in `affected`.

        A fact (x,z) is witnessed by each remaining edge E(y,z) entering z with
        y == x (the length-one edge) or with (x,y) still holding.  An affected
        fact holds only once it is itself restored (restoration is per-source x,
        so another source's restored fact never counts); a fact outside the
        affected set holds iff its closure row still exists.  The restore set
        grows to a fixed point so facts that witness one another inside a
        strongly connected component can bootstrap; supports are then counted
        once every witness is known.  Facts that never restore keep the zero
        support (absent row) written during over-deletion.
        """
        def holds(x, y):
            pair = (x, y)
            if pair in affected:
                return pair in restored
            return self.store.support(x, y) > 0

        restored = set()
        pending = set(affected)
        while pending:
            newly = set()
            for x, z in sorted(pending):
                for y in self.store.incoming(z):
                    if y == x or holds(x, y):
                        newly.add((x, z))
                        break
            if not newly:
                break
            restored |= newly
            pending -= newly

        for x, z in restored:
            support = 0
            for y in self.store.incoming(z):
                if y == x or holds(x, y):
                    support += 1
            self.store.set_support(x, z, support)

    # ------------------------------------------------------------------ insert
    def insert(self, src, dst):
        if self.store.has_edge(src, dst):
            return
        # Every x reaching src (src included) gains paths that enter the new
        # edge E(src,dst); the only facts that may change are (x,w) with w
        # reachable from dst.  Read both index ranges while the edge is absent so
        # they describe only pre-existing paths.
        roots = {src}
        roots.update(self.store.predecessors(src))
        cone = {dst}
        cone.update(self.store.successors(dst))
        self.store.add_edge(src, dst)

        # Over-delete the local cone, then re-derive exact supports.  Every
        # affected fact still has a path through the new edge, so presence is
        # unchanged; this both admits newly reachable facts and repairs counts
        # that cyclic propagation would otherwise inflate.
        affected = {(x, w) for x in roots for w in cone}
        for x, w in affected:
            self.store.set_support(x, w, 0)
        self._rederive(affected)

    # ------------------------------------------------------------------ delete
    def delete(self, src, dst):
        if not self.store.has_edge(src, dst):
            return
        # Over-delete seed: every x reaching src (including src itself) could
        # use E(src,dst) as the final witness of (x,dst).  Read predecessors
        # while the edge still exists, then remove it so expansion follows only
        # the surviving graph.
        candidates = {(x, dst) for x in self.store.predecessors(src)}
        candidates.add((src, dst))
        self.store.remove_edge(src, dst)

        # Phase 1, local over-deletion.  Removing (x,z) can invalidate only
        # relations (x,w) whose final witness goes through z, so expansion stays
        # within the same source x and touches just closure facts actually
        # marked.  Relations outside this set cannot have used the removed edge
        # and are left untouched (stable branches are never re-walked).
        removed = set()
        pending = deque(candidates)
        while pending:
            x, z = pending.popleft()
            if (x, z) in removed:
                continue
            removed.add((x, z))
            self.store.set_support(x, z, 0)
            for w in self.store.successors(z):
                if (x, w) not in removed and self.store.support(x, w) > 0:
                    pending.append((x, w))

        # Phase 2: restore the marked facts that still have an alternative path.
        self._rederive(removed)

    def apply(self, add, remove):
        for src, dst in sorted(remove):
            self.delete(src, dst)
        for src, dst in sorted(add):
            self.insert(src, dst)

"""Target-swap operators extracted from the Concerto rebuttal implementation.

The grouped single-group fallback is the protocol described in the paper.
The original 32-attempt derangement fallback is replaced by an explicit error.
"""

import torch


class TargetSwapOperator:
    def __init__(self):
        self._shortcut_probe_logs = set()

    @staticmethod
    def _sample_derangement(num_items, device):
        if num_items <= 1:
            raise ValueError(
                "cross_scene_target_swap requires at least two scenes with valid teacher targets per batch."
            )
        identity = torch.arange(num_items, device=device)
        for _ in range(32):
            perm = torch.randperm(num_items, device=device)
            if not torch.any(perm == identity):
                return perm
        raise RuntimeError("failed to sample a derangement in 32 attempts")

    @staticmethod
    def _resample_rows_with_replacement(rows, target_length):
        if rows.shape[0] == 0:
            raise ValueError("Cannot resample teacher targets from an empty source block.")
        if rows.shape[0] == target_length:
            return rows
        sample_index = torch.randint(
            rows.shape[0], (target_length,), device=rows.device
        )
        return rows[sample_index]

    def _log_shortcut_probe_once(self, tag, message):
        if tag not in self._shortcut_probe_logs:
            self._shortcut_probe_logs.add(tag)
            print(message)

    def _apply_global_target_permutation(
        self, teacher_target, mode_name="global_target_permutation"
    ):
        if teacher_target.shape[0] <= 1:
            self._log_shortcut_probe_once(
                f"{mode_name}:skip",
                f"[shortcut_probe] {mode_name} skip=identity rows={teacher_target.shape[0]}",
            )
            return teacher_target
        perm = self._sample_derangement(teacher_target.shape[0], teacher_target.device)
        self._log_shortcut_probe_once(
            mode_name,
            f"[shortcut_probe] {mode_name} self_assignments=0 rows={teacher_target.shape[0]}",
        )
        return teacher_target[perm]

    def _apply_group_target_swap(self, teacher_target, target_group_index, mode_name):
        unique_group_index = torch.unique(target_group_index, sorted=True)
        if unique_group_index.numel() <= 1:
            self._log_shortcut_probe_once(
                f"{mode_name}:fallback",
                f"[shortcut_probe] {mode_name} fallback=global_target_permutation valid_groups=1",
            )
            return self._apply_global_target_permutation(
                teacher_target, mode_name=f"{mode_name}:fallback_permutation"
            )

        perm = self._sample_derangement(unique_group_index.numel(), teacher_target.device)
        swapped = teacher_target.clone()
        swap_log_parts = []
        for dest_pos, src_pos in enumerate(perm.tolist()):
            dest_group = unique_group_index[dest_pos]
            src_group = unique_group_index[src_pos]
            dest_mask = target_group_index == dest_group
            source_rows = teacher_target[target_group_index == src_group]
            dest_count = int(dest_mask.sum().item())
            swapped[dest_mask] = self._resample_rows_with_replacement(
                source_rows, dest_count
            )
            swap_log_parts.append(
                f"{int(dest_group.item())}<-{int(src_group.item())}:{dest_count}/{int(source_rows.shape[0])}"
            )

        self._log_shortcut_probe_once(
            mode_name,
            f"[shortcut_probe] {mode_name} self_assignments=0 mapping={', '.join(swap_log_parts)}",
        )
        return swapped

    def _apply_cross_scene_target_swap(self, teacher_target, target_batch_index):
        return self._apply_group_target_swap(
            teacher_target,
            target_batch_index,
            mode_name="cross_scene_target_swap",
        )

    def _apply_cross_image_target_swap(self, teacher_target, target_image_index):
        return self._apply_group_target_swap(
            teacher_target,
            target_image_index,
            mode_name="cross_image_target_swap",
        )

    def apply(self, teacher_target, mode, group_index):
        if mode == "none":
            return teacher_target
        if mode == "global_target_permutation":
            return self._apply_global_target_permutation(teacher_target)
        if group_index is None:
            raise ValueError("group_index is required for a grouped swap")
        if mode == "cross_image_target_swap":
            return self._apply_cross_image_target_swap(teacher_target, group_index)
        if mode == "cross_scene_target_swap":
            return self._apply_cross_scene_target_swap(teacher_target, group_index)
        raise ValueError(f"unknown target swap mode: {mode}")

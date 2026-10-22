"""Convergence-aware finite-difference diagnostics, not a paper ranking test."""
import torch


@torch.no_grad()
def check_derivative(render, t, h, atol=1e-3, rtol=.1):
    """Compare the analytic tangent with a decreasing-step central difference.

    Report all moving pixels AND the FD-converged subset. A lack of converged
    pixels is inconclusive, not a pass and not proof the tangent is wrong.
    Contributor counts are a conservative diagnostic, not a complete branch tape.
    """
    if h <= 0:
        raise ValueError('h must be positive')
    center = render(t)
    an = center.L_dot.double()
    fds, branch_stable = [], center.rendered.clone()
    for step in (h, h/2, h/4):
        plus, minus = render(t+step), render(t-step)
        # Subtract in float64; float32 primal rounding is still present.
        fds.append((plus.L.double()-minus.L.double())/(2*step))
        branch_stable &= (plus.c == center.c) & (minus.c == center.c)
        branch_stable &= plus.rendered & minus.rendered
    fd = fds[-1]
    finite = torch.isfinite(an) & torch.isfinite(fd) & center.rendered
    magnitude = torch.maximum(an.abs(), fd.abs())
    valid_magnitude = magnitude[finite]
    if valid_magnitude.numel() == 0:
        return dict(status='FAIL', reason='No finite rendered derivatives', time_s=t, h_s=h)
    moving = finite & (magnitude > max(atol, .05*float(valid_magnitude.max())))
    if not bool(moving.any()):
        return dict(status='INCONCLUSIVE', reason='No moving pixels', time_s=t, h_s=h)
    scale = magnitude.clamp_min(atol)
    fd_change = (fds[-1]-fds[-2]).abs()
    converged = moving & branch_stable & (fd_change <= atol + .05*scale)

    def summarize(mask):
        if not bool(mask.any()):
            return dict(pixels=0, median_relative_error=None, fraction_within_tolerance=None)
        error = (an-fd).abs()[mask]
        return dict(pixels=int(mask.sum()), median_relative_error=float((error/scale[mask]).median()),
                    fraction_within_tolerance=float((error <= atol+rtol*scale[mask]).double().mean()))

    full, stable = summarize(moving), summarize(converged)
    fraction = int(converged.sum()) / int(moving.sum())
    status = 'INCONCLUSIVE'
    if fraction >= .9:
        status = 'PASS' if stable['median_relative_error'] < .05 and stable['fraction_within_tolerance'] > .9 else 'FAIL'
    if bool((center.rendered & ~finite).any()):
        status = 'FAIL'
    return dict(status=status, time_s=float(t), h_s=float(h), atol=atol, rtol=rtol,
                all_moving=full, fd_converged=stable, converged_fraction=fraction,
                contributor_stable_fraction=float(branch_stable[moving].double().mean()),
                fd_step_change_median=float((fd_change/scale)[moving].median()),
                coarse_fd_step_change_median=float(((fds[1]-fds[0]).abs()/scale)[moving].median()),
                nonfinite_rendered_pixels=int((center.rendered & ~finite).sum()),
                limitation='FD convergence and equal contributor counts do not guarantee identical visibility branches.')

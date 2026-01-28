from __future__ import print_function
import sys
import numpy as np
import logging

###########################################################################################################################
# TABLES
###########################################################################################################################

def get_limit(samples, param, limit=1, expected_marker="=", get_value=True, both=False):
    """
    Extract LaTeX limits for a parameter directly from an MCSamples object.
    """

    def _inline(lim):
        return samples.getInlineLatex(param, limit=lim)

    if get_value is True:

        if both is False:
            inline = _inline(limit)

            if expected_marker in inline:
                return inline.split(expected_marker, 1)[1]
            else:
                expected_marker = "<"
                if expected_marker in inline:
                    return "<" + inline.split(expected_marker, 1)[1]
                expected_marker = ">"
                if expected_marker in inline:
                    return ">" + inline.split(expected_marker, 1)[1]
                return inline  # fallback

        # both=True -> 68% (limit=1) and 95% (limit=2)
        in1 = _inline(1)
        in2 = _inline(2)

        if "=" in in1:
            v1 = in1.split("=", 1)[1]
        elif "<" in in1:
            v1 = "<" + in1.split("<", 1)[1]
        elif ">" in in1:
            v1 = ">" + in1.split(">", 1)[1]
        else:
            v1 = in1

        if "=" in in2:
            v2 = in2.split("=", 1)[1]
        elif "<" in in2:
            v2 = "<" + in2.split("<", 1)[1]
        elif ">" in in2:
            v2 = ">" + in2.split(">", 1)[1]
        else:
            v2 = in2

        return v1 + "\\, (" + v2 + " )"

    else:
        inline = _inline(limit)

        if expected_marker in inline:
            return inline.split(expected_marker, 1)[0]
        expected_marker = "<"
        if expected_marker in inline:
            return inline.split(expected_marker, 1)[0]
        expected_marker = ">"
        if expected_marker in inline:
            return inline.split(expected_marker, 1)[0]
        return inline  # fallback


def get_limits_for_param(samples_list, param, limit=1):
    """
    Print a LaTeX table row with parameter limits for a list of MCSamples.
    """

    both = False

    if limit in ("both", "Both", "b", "B"):
        limit = 1
        both = True
    else:
        if str(limit) in ("1", "2", "3"):
            limit = int(limit)
        else:
            logging.error("limit must be: 1 (68 CL) | 2 (95 CL) | 3 | both (68 CL + 95 CL)")
            logging.error("Use: parameter:limit (e.g. ns:1)")
            sys.exit(1)

    # Parameter label from the first sample
    print("$", get_limit(samples_list[0], param, get_value=False), "$", "&", end=" ")

    for i, s in enumerate(samples_list):
        if i == len(samples_list) - 1:
            print("$", get_limit(s, param, limit=limit, both=both), "$", "\\\\")
        else:
            print("$", get_limit(s, param, limit=limit, both=both), "$", "&", end=" ")


def _get_param_names(samples):
    """
    Return parameter names from an MCSamples object in a version-robust way.
    """
    try:
        return samples.getParamNames().list()
    except Exception:
        try:
            return [p.name for p in samples.getParamNames().names]
        except Exception:
            return list(samples.paramNames.names)


def _chi2_component_filter(name, exclude_exact=('chi2__CMB', 'chi2__BAO', 'chi2__SN')):
    """
    Keep only chi2__* components, excluding aggregate groups (CMB/BAO/SN).
    """
    if not name.startswith('chi2__'):
        return False
    if name == 'chi2__':
        return False
    if name in exclude_exact:
        return False
    return True


def get_chi2_statistics(samples, rtol=1e-4, atol=1e-4, print_all_components=True, header=None, silent=False):
    """
    Print chi2 diagnostics at the minimum chi2 point for a given MCSamples object.
    If silent=True, do not print anything, only return the dictionary.
    """

    def _p(*args, **kwargs):
        if not silent:
            print(*args, **kwargs)

    names = _get_param_names(samples)

    if 'chi2' not in names:
        _p('-----------------------------------------------')
        if header is not None:
            _p('Sample:', header)
        _p('WARNING: parameter "chi2" not found')
        _p('Available chi2-like parameters:', [p for p in names if p.startswith('chi2')])
        _p('-----------------------------------------------')
        return None

    chi2_arr = samples['chi2']
    imin = int(np.argmin(chi2_arr))
    chi2_min = float(chi2_arr[imin])

    single_components = sorted([n for n in names if _chi2_component_filter(n)])
    all_chi2_components = sorted([n for n in names if n.startswith('chi2__') and n != 'chi2__'])

    _p('-----------------------------------------------')
    if header is not None:
        _p('Sample:', header)
    _p('min(chi2) =', chi2_min)
    _p('-----------------------------------------------')

    if len(single_components) == 0:
        _p('No individual chi2__ components found after filtering.')
        if print_all_components and len(all_chi2_components) > 0:
            _p('Found the following chi2__ entries:')
            for n in all_chi2_components:
                try:
                    _p(' ', n, float(samples[n][imin]))
                except Exception:
                    pass
        _p('-----------------------------------------------')
        return {'chi2_min': chi2_min, 'imin': imin, 'single_components': {}, 'sum_single': np.nan}

    sum_single = 0.0
    single_dict = {}
    for n in single_components:
        v = float(samples[n][imin])
        single_dict[n] = v
        sum_single += v
        _p(n, v)

    if print_all_components:
        groups_to_show = [g for g in all_chi2_components if g in ('chi2__CMB', 'chi2__BAO', 'chi2__SN')]
        if len(groups_to_show) > 0:
            _p('-----------------------------------------------')
            _p('Aggregate likelihood groups (excluded from the check):')
            for g in groups_to_show:
                try:
                    _p(g, float(samples[g][imin]))
                except Exception:
                    pass

    _p('-----------------------------------------------')
    if np.isclose(sum_single, chi2_min, rtol=rtol, atol=atol):
        _p('Verified: sum of individual chi2 components matches total chi2.')
    else:
        _p('Mismatch (often harmless): chi2_min =', chi2_min, 'vs sum of components =', sum_single)
    _p('-----------------------------------------------')

    return {'chi2_min': chi2_min, 'imin': imin, 'single_components': single_dict, 'sum_single': sum_single}


def get_chi2_row_for_table(samples_list, label=r"$\chi^2_{\min}$", headers=None):
    """
    Print a LaTeX table row with chi2_min for each MCSamples object (no diagnostics).
    """
    print(label, "&", end=" ")
    for i, s in enumerate(samples_list):
        out = get_chi2_statistics(
            s,
            print_all_components=False,
            header=(headers[i] if headers else None),
            silent=True
        )
        val = out['chi2_min'] if out is not None else np.nan

        if i == len(samples_list) - 1:
            print(f"${val:.3f}$", "\\\\")
        else:
            print(f"${val:.3f}$", "&", end=" ")


def get_chi2_component_rows_for_table(samples_list, headers=None, label_prefix="", exclude_exact=('chi2__CMB', 'chi2__BAO', 'chi2__SN')):
    """
    Print LaTeX table rows for each individual chi2__ component (excluding aggregate groups).
    Each row is evaluated at the minimum-chi2 point of each sample.
    """

    # Collect union of chi2__ component names across all samples
    all_components = set()
    for s in samples_list:
        names = _get_param_names(s)
        for n in names:
            if n.startswith('chi2__') and n != 'chi2__' and (n not in exclude_exact):
                all_components.add(n)

    all_components = sorted(all_components)

    # Print one row per component
    for comp in all_components:
        # Row label: chi2__X -> chi2_X
        row_label = comp.replace("chi2__", "chi2_")
        if label_prefix:
            row_label = label_prefix + row_label

        print(row_label, "&", end=" ")

        for i, s in enumerate(samples_list):
            out = get_chi2_statistics(
                s,
                print_all_components=False,
                header=(headers[i] if headers else None),
                silent=True
            )
            imin = out['imin'] if out is not None else None

            # If the component is missing in this chain, print a dash
            if imin is None:
                val = None
            else:
                names = _get_param_names(s)
                if comp in names:
                    val = float(s[comp][imin])
                else:
                    val = None

            if i == len(samples_list) - 1:
                if val is None or (isinstance(val, float) and not np.isfinite(val)):
                    print("$-$", "\\\\")
                else:
                    print(f"${val:.3f}$", "\\\\")
            else:
                if val is None or (isinstance(val, float) and not np.isfinite(val)):
                    print("$-$", "&", end=" ")
                else:
                    print(f"${val:.3f}$", "&", end=" ")


def get_table(samples_list, params, col_labels=False, chi2=False, caption="Caption TBW", label="tab.label", info=False, headers_for_chi2=None):
    """
    Generate a LaTeX table from a list of MCSamples objects.
    """

    if info is True:
        print("Table requested with the following configuration\n")
        print("Number of columns:", len(samples_list) + 1)

        if col_labels is False:
            print("Column labels: not provided")
        else:
            print("Column labels:", "| Parameter |", *col_labels)

        print("Caption:", caption)
        print("Table label:", label)
        print("\nParameters:")
        for p in params:
            print(" ", p)
        print("\n")

    print("Please copy the following block into a LaTeX document\n")
    print("%====================================================\n")

    print("\\begin{table*}")
    print("\\begin{center}")
    print("\\renewcommand{\\arraystretch}{1.5}")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{l " + " c" * len(samples_list) + "}")
    print("\\hline")

    if col_labels is not False:
        print("\\textbf{Parameter} &", end=" ")
        for i, lab in enumerate(col_labels):
            if i == len(col_labels) - 1:
                print("\\textbf{", lab, "}", "\\\\")
            else:
                print("\\textbf{", lab, "}", "&", end=" ")
        print("\\hline\\hline")

    print("")

    for p in params:
        pname = p.split(":")[0]
        plim = p.split(":", 1)[1]
        get_limits_for_param(samples_list, pname, limit=plim)

    if chi2 is not False:
        get_chi2_row_for_table(samples_list, headers=headers_for_chi2)
        print("%===================== Chi2 Components (erase if not needed) =============================")
        get_chi2_component_rows_for_table(samples_list, headers=headers_for_chi2)
        print("%========================================================================================")

    print("")
    print("\\hline \\hline")
    print("\\end{tabular} }")
    print("\\end{center}")
    print("\\caption{", caption, "}")
    print("\\label{", label, "}")
    print("\\end{table*}")
    print("\n%====================================================")
    if chi2 is not False:
        print("")
        print("")
        print("\n%====================================================")
        print("% chi2 diagnostics at the minimum chi2 point")
        print("% Aggregate likelihoods (CMB/BAO/SN) are excluded from the consistency check")
        print("%====================================================\n")
        for i, s in enumerate(samples_list):
            get_chi2_statistics(s, header=(headers_for_chi2[i] if headers_for_chi2 else None))
            
    return

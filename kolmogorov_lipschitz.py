"""
Kolmogorov Lipschitz Function Algorithm
====================================================================================

Optimized implementation preserving Jonas Actor's original algorithm
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import sys
from operator import attrgetter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from intervaltree import Interval, IntervalTree
import time
import itertools as itt
import gmpy2
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(
    description='Lipschitz Functions',
    epilog='MPFR + NumPy implementation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dim', type=int, default=2, help='The spatial dimension')
parser.add_argument('--J', type=int, default=8, help='Number of iterations')
parser.add_argument('--break_ratio', type=float, default=0.5, help='Reduction in size of largest town, at each step')
parser.add_argument('--draw', type=int, default=0, help='Use --draw 1 to draw the intervals after every iteration, and --draw 2 to draw only the end result')
parser.add_argument('--plot', type=int, default=0, help='Use --plot 1 to plot the resulting function')
parser.add_argument('--Phi', type=int, default=0, help="Full Psi plot")
parser.add_argument('--prec', type=int, default=2**8, help='Precision in bits')
parser.add_argument('--verbose', type=int, default=0, help='Prints diagnostic output to screen')
parser.add_argument('--summary', type=int, default=1, help='Summary of town and function construction displayed after each iteration')

args = parser.parse_args()

# Configure MPFR precision
gmpy2.get_context().precision = args.prec
print(f"MPFR precision set to {args.prec} bits")

def mpfr(value):
    """Create MPFR number - direct replacement for mp.mpf"""
    return gmpy2.mpfr(str(value))

class Town(object):
    def __init__(self, start, end, val, nv, birth, parent):
        self.end = end
        self.start = start
        self.length = mpfr(str(end-start))  # MPFR instead of mp.mpf
        self.center = start + self.length / 2.0
        self.val = val
        self.vleft = val
        self.vright = val
        self.nextval = nv
        self.birth = birth
        self.parent = parent
        self.children = []

    def __lt__(self, other):
        return self.start < other.start

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.start == other.start) and (self.end == other.end)

    def __hash__(self):
        return hash((self.start, self.length, self.val))

    def __repr__(self):
        return str(self.start) + "~~~" + str(self.end)

    def __str__(self):
        return str(self.start) + "~~~" + str(self.end)

class Gap(object):
    def __init__(self, start, end, parent, vall, valr):
        self.start = start
        self.end = end
        self.length = end - start
        self.parent = parent
        self.children = []
        self.vleft = vall
        self.vright = valr

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        return str(self.start) + "~~~" + str(self.end)

    def __str__(self):
        return str(self.start) + "~~~" + str(self.end)

class Hole(object):
    def __init__(self, start, end, valleft, valright, pts, shifts, tl):
        self.start = start
        self.end = end
        self.length = end - start
        self.valleft = valleft
        self.valright = valright
        self.townleft = tl
        try:
            _ = (e for e in pts)
            self.pts = pts
            self.shifts = shifts
        except TypeError:
            self.pts = [pts]
            self.shifts = [shifts]

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        return str(self.start) + "~~~" + str(self.pts) + "~~~" + str(self.end)

    def __str__(self):
        return str(self.start) + "~~~" + str(self.pts) + "~~~" + str(self.end)

class BT(object):
    def __init__(self, treetop):
        self.levels = mpfr('1')  # MPFR instead of mp.mpf
        self.top = treetop
        self.leaves = IntervalTree()
        self.leafTowns = IntervalTree()
        self.leafGaps = IntervalTree()

# Global parameters
n = args.dim
J = args.J
Q = mpfr(2*n + 1)  # MPFR instead of mp.mpf
Qint = 2*n + 1
P = mpfr(n)  # MPFR instead of mp.mpf
theta = args.break_ratio
epsilon = mpfr('1') / (Qint - 1)  # MPFR arithmetic
draw = args.draw
reftown = Town(0, 0, 0, None, 0, None)
verbose = args.verbose
summary = args.summary
plotfxn = args.plot
plotPhi = args.Phi

def evaluate(B, pt, q):
    if pt >= 1.0:
        pt = mpfr('1') - mpfr('2')**(-J - 2)  # MPFR arithmetic
    pt = pt - q*epsilon
    bb = B.leaves[pt]
    [b] = bb
    bl = b.data.vleft
    br = b.data.vright
    bs = b.data.start
    be = b.data.end
    return bl + (br-bl) / (be-bs) * (pt-bs)

def Phi(B, x, q):
    return sum([(mpfr(1.0) / ((p+2)**mpfr(0.5))) * evaluate(B, x[p], q) for p in range(len(x))])

def plotfxn_func(town_tree, shift):
    points = []
    vals = []
    st = sorted(list(town_tree))
    for twn in st:
        t = twn.data
        if t.start + shift >= 0.0 and t.end + shift <= 1.0:
            points.append(float(t.start + shift))
            points.append(float(t.end + shift))
            vals.append(float(t.val))
            vals.append(float(t.val))
        elif t.start + shift < 1.0 and t.end + shift > 1.0:
            points.append(float(t.start + shift))
            points.append(1.0)
            vals.append(float(t.val))
            vals.append(float(t.val))
        elif t.start + shift < 0.0 and t.end + shift > 0.0:
            points.append(0.0)
            points.append(float(t.end + shift))
            vals.append(float(t.val))
            vals.append(float(t.val))
    plt.plot(points, vals, '-')

def plottowns(town_tree):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    yshift = 0.05
    for q in range(Qint):
        for twn in town_tree:
            t = twn.data
            ax1.add_patch(
                patches.Rectangle(
                    (float(t.start + q*epsilon), yshift + q * 0.2),
                    float(t.length),
                    0.1,
                )
            )
    plt.xlim([-1, 2])
    plt.show()

def drawBTNode(node, vshift, yloc, ax1):
    if type(node) == type(reftown):
        ax1.add_patch(
            patches.Rectangle(
                (node.start, yloc),
                node.end - node.start,
                vshift,
            )
        )

def visualizeBTTree(node, level, vshift, ax1):
    drawBTNode(node, vshift, vshift*(2*level + 1), ax1)
    for c in node.children:
        visualizeBTTree(c, level+1, vshift, ax1)

def visualizeBT(B):
    BT_obj = B.top
    numlevels = B.levels
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    vshift = 1.0 / (2.0 * numlevels + 1.0)
    visualizeBTTree(BT_obj, 0, vshift, ax1)
    plt.xlim([-1, 1])
    plt.show()

def get_towns_above_thresh(town_tree, thresh):
    return [t.data for t in town_tree if t.data.length > thresh]

def get_my_town(town_tree, pt):
    town_ret = town_tree[pt]
    emptyset = set()
    if town_ret != emptyset:
        [tr] = town_ret
        if tr.data.start == pt:
            return None
        elif tr.data.end == pt:
            return None
        else:
            return tr.data
    else:
        return None

def is_pt_in_town(town_tree, pt, shift):
    if pt - shift <= -1:
        return "before"
    elif pt - shift >= 1:
        return "after"
    else:
        town_ret = get_my_town(town_tree, pt - shift)
        if town_ret == None:
            return 'hole'
        else:
            return town_ret

def get_gap(town_tree, breakpts, pt):
    rhol = np.inf
    rhor = np.inf
    if verbose:
        print("\nGetting break for point", pt, "\n")
    for q in range(-Qint+1, Qint):
        my_town = get_my_town(town_tree, pt + q*epsilon)
        if my_town is not None:
            if verbose:
                print(pt + q*epsilon, ":\t", my_town, "\t\trhol:\t", pt + q*epsilon - my_town.start, "\trhor:\t", my_town.end - (pt + q*epsilon))
            dl = (mpfr(2) / 3) * (pt + q*epsilon - my_town.start)  # MPFR arithmetic
            if dl < rhol:
                rhol = dl
            dr = (mpfr(2) / 3) * (my_town.end - (pt + q*epsilon))  # MPFR arithmetic
            if dr < rhor:
                rhor = dr
        else:
            if verbose:
                print(pt + q*epsilon, ":\t", my_town)
    for q in range(-(2*Qint)+1, 2*Qint):
        rhoptsl = [pt - (bp+q*epsilon) for bp in breakpts if pt > bp + q*epsilon]
        rhoptsr = [(bp+q*epsilon) - pt for bp in breakpts if pt < bp + q*epsilon]
        if len(rhoptsl) > 0:
            dl = (mpfr(1) / 3) * min(rhoptsl)  # MPFR arithmetic
            if dl < rhol:
                rhol = dl
        if len(rhoptsr) > 0:
            dr = (mpfr(1) / 3) * min(rhoptsr)  # MPFR arithmetic
            if dr < rhor:
                rhor = dr
    assert rhol > 0
    assert rhor > 0
    break_start = pt - min(rhol, (mpfr(1) / 3) * epsilon)  # MPFR arithmetic
    break_end = pt + min(rhor, (mpfr(1) / 3) * epsilon)    # MPFR arithmetic
    if verbose:
        print('\nbreak:', break_start, "~~~", pt, '~~~', break_end, "\n")
    return break_start, break_end

def get_border_towns(town_tree, pt, town_list):
    QQ = list(range(-Qint+1, Qint))
    qq = [q for q, t in enumerate(town_list) if t=='hole']
    tleft = []
    tright = []
    shifts = []
    for p in range(len(qq)):
        qk = -QQ[qq[p]]
        shifts.append(qk * epsilon)
        tleft.append(max([t.data for t in town_tree if t.data.end < pt + shifts[p]], key=attrgetter('end')))
        tright.append(min([t.data for t in town_tree if t.data.start > pt + shifts[p]], key=attrgetter('start')))
    return tleft, tright, shifts

def refine(j, B):
    timestart = time.time()
    newLeaves = IntervalTree()
    newLeafTowns = IntervalTree()
    newLeafGaps = IntervalTree()
    town_tree = B.leafTowns

    towns_to_break = get_towns_above_thresh(B.leafTowns, theta**j)
    ntb = len(towns_to_break)
    if verbose:
        print('ntb', ntb)
        print()
    if ntb < 1:
        if verbose:
            print('No towns to break at this iteration')
        return

    # PLUGS section
    hole_tree = IntervalTree()
    for tb in towns_to_break:
        pt = tb.center
        if verbose:
            print()
            print('tb', tb)
            print('pt', pt)
        pt_towns = [None] * (2*Qint - 1)
        for q in range(-Qint+1, Qint):
            pt_towns[q+Qint-1] = is_pt_in_town(town_tree, pt, q*epsilon)
        num_h = sum(t=='hole' for t in pt_towns)
        if verbose:
            print('pt_towns', pt_towns)
            print('nh', num_h)
        if num_h > 2:
            print("NOT RIGHT NUMBER OF BREAKS")
            return
        if num_h > 0:
            gap_left, gap_right, shifts = get_border_towns(town_tree, pt, pt_towns)
            for i in range(num_h):
                already_present = hole_tree[gap_left[i].end]
                if len(already_present) == 0:
                    h = Hole(gap_left[i].end, gap_right[i].start, gap_left[i].val, gap_right[i].val, pt, shifts[i], gap_left[i])
                    hole_tree.addi(gap_left[i].end, gap_right[i].start, h)
                    if verbose:
                        print("Hole:\t", h)
                else:
                    h = already_present.pop().data
                    h.pts.append(pt)
                    h.shifts.append(shifts[i])
                    hole_tree.addi(h.start, h.end, h)
                    if verbose:
                        print("Hole:\t", h)

    for g in B.leafGaps:
        gb = g.data
        ht_intersect = hole_tree[gb.start:gb.end]
        if ht_intersect == set():
            gb_dup = Gap(gb.start, gb.end, gb, gb.vleft, gb.vright)
            gb.children = [gb_dup]
            newLeaves.addi(gb_dup.start, gb_dup.end, gb_dup)
            newLeafGaps.addi(gb_dup.start, gb_dup.end, gb_dup)
        else:
            assert len(ht_intersect) == 1
            [w] = ht_intersect
            h = w.data
            num_pts = len(h.pts)
            if verbose:
                print('\nHole:\t', h)
            old_slope = (h.valright - h.valleft) / h.length
            new_slope = mpfr(1) - mpfr(2)**(-j-1)  # MPFR arithmetic

            p_shifted = [h.pts[p] + h.shifts[p] for p in range(num_pts)]
            p_shifted.sort()
            f = [h.valleft + old_slope * (p_shifted[p] - h.start) for p in range(num_pts)]
            f.insert(0, h.valleft)
            f.append(h.valright)

            # Matrix operations - NumPy replacement for mpmath
            C = np.zeros((2*num_pts, 2*num_pts), dtype=np.float64)
            z = np.zeros(2*num_pts, dtype=np.float64)
            for i in range(num_pts + 1):
                if i == 0:
                    C[0, 0] = float(new_slope)
                    z[0] = float(f[1] - f[0] + new_slope * h.start)
                elif i == num_pts:
                    C[num_pts, 2*num_pts-1] = float(-new_slope)
                    z[num_pts] = float(f[-1] - f[-2] - new_slope * h.end)
                else:
                    C[i, 2*i-1] = float(-new_slope)
                    C[i, 2*i] = float(new_slope)
                    z[i] = float(f[i+1] - f[i])
            if num_pts > 1:
                for i in range(num_pts+1, 2*num_pts):
                    C[i, 2*(i-num_pts)-1] = float(mpfr(1))
                    C[i, 2*(i-num_pts)] = float(mpfr(1))
                    z[i] = float(p_shifted[i-num_pts-1] + p_shifted[i-num_pts])

            if verbose:
                print('System: Cx = z')
                print('C:\n', C)
                print('z:\n', z, '\n')

            x = np.linalg.solve(C, z)

            for i in range(num_pts):
                # Convert NumPy results back to MPFR
                plug_l = mpfr(str(x[2*i]))
                plug_r = mpfr(str(x[2*i + 1]))
                if verbose:
                    print('Point:\t', h.pts[i])
                    print("hole_start:\t", h.start)
                    print("plug_l:\t", plug_l)
                    print("pt_shifted:\t", p_shifted[i])
                    print("plug_r:\t", plug_r)
                    print("hole_end:\t", h.end)
                plug = Town(plug_l, plug_r, f[i+1], f[i+2], j, gb)
                town_tree.addi(plug_l, plug_r, plug)
                newLeaves.addi(plug_l, plug_r, plug)
                newLeafTowns.addi(plug_l, plug_r, plug)

                if i == 0:
                    lg_l = h.start
                else:
                    lg_l = mpfr(str(x[2*i - 1]))
                left_gap = Gap(lg_l, plug_l, gb, f[i], f[i+1])
                newLeaves.addi(left_gap.start, left_gap.end, left_gap)
                newLeafGaps.addi(left_gap.start, left_gap.end, left_gap)

                gb.children = gb.children + [plug, left_gap]

            right_gap = Gap(plug_r, h.end, gb, f[-2], f[-1])
            newLeaves.addi(right_gap.start, right_gap.end, right_gap)
            newLeafGaps.addi(right_gap.start, right_gap.end, right_gap)
            gb.children = gb.children + [right_gap]

            h.townleft.nextval = f[1]

    if verbose:
        print("\nPlugged Towntree:")
        print([t.data for t in town_tree])
        print()

    # GAPS section
    breakpts = [tb.center for tb in towns_to_break]
    for t in B.leafTowns:
        tb = t.data
        if tb.length > theta**j:
            pt = tb.center
            break_start, break_end = get_gap(town_tree, breakpts, pt)
            if (tb.nextval == None) or ((tb.nextval - tb.val) > (break_end - break_start) * mpfr('0.5')):  # MPFR arithmetic
                new_val = tb.val + (break_end - break_start) * mpfr('0.5')  # MPFR arithmetic
            else:
                new_val = (tb.val + tb.nextval) * mpfr('0.5')  # MPFR arithmetic

            left = Town(tb.start, break_start, tb.val, new_val, tb.birth, tb)
            right = Town(break_end, tb.end, new_val, tb.nextval, tb.birth, tb)
            middle = Gap(break_start, break_end, tb, tb.val, new_val)

            tb.children = [left, right, middle]
            newLeaves.addi(left.start, left.end, left)
            newLeaves.addi(right.start, right.end, right)
            newLeaves.addi(middle.start, middle.end, middle)
            newLeafTowns.addi(left.start, left.end, left)
            newLeafTowns.addi(right.start, right.end, right)
            newLeafGaps.addi(middle.start, middle.end, middle)
        else:
            tb_dup = Town(tb.start, tb.end, tb.val, tb.nextval, tb.birth+1, tb)
            tb.children = [tb_dup]
            newLeaves.addi(tb_dup.start, tb_dup.end, tb_dup)
            newLeafTowns.addi(tb_dup.start, tb_dup.end, tb_dup)

    # UPDATE B
    B.levels = B.levels + 1
    B.leaves = newLeaves
    B.leafTowns = newLeafTowns
    B.leafGaps = newLeafGaps

    time_end = time.time()
    if summary or verbose:
        print('\ntheta:\t\t\t', theta**j)
        print('smallest town size:\t', min([t.data.length for t in B.leafTowns]))
        print('largest town size:\t', max([t.data.length for t in B.leafTowns]))
        print('number of towns:\t', len(B.leafTowns))
        print('number of towns broken:\t', ntb)
        print('total length:\t\t', sum([t.data.length for t in B.leafTowns]))
        print('time elapsed:\t\t', time_end - timestart)
        print()

    if draw == 1:
        plottowns(B.leafTowns)

def fullRefine(J):
    print()
    print("Setting up Sprecher Town System")

    domain = IntervalTree()
    start = mpfr(-1)  # MPFR instead of mp.mpf
    end = mpfr(1)     # MPFR instead of mp.mpf
    I = Town(start, end, mpfr(0), None, 0, None)  # MPFR instead of mp.mpf
    domain.addi(start, end, I)

    B = BT(I)
    B.leaves = domain
    B.leafTowns = domain

    print('Beginning refinement...')
    print()

    j = mpfr('0')  # MPFR instead of mp.mpf
    while j < J:
        if summary or verbose:
            print('Beginning level', j)
        refine(j, B)
        j = j+1

    return B

def plotPhi_func(B):
    assert P == 2
    print("Creating 3D Phi surface plots...")

    for q in range(int(Q)):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create grid for surface plot
        x_range = np.linspace(0.0, 0.99999, 50)
        y_range = np.linspace(0.0, 0.99999, 50)
        X, Y = np.meshgrid(x_range, y_range)

        # Calculate Phi values
        Z = np.zeros_like(X)
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                try:
                    Z[j, i] = float(Phi(B, (X[j, i], Y[j, i]), mpfr(str(q))))
                except:
                    Z[j, i] = 0  # Handle edge cases

        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(f'Φ_{q}(x,y)')
        ax.set_title(f'Phi Surface Plot - q={q}')

        # Add colorbar
        fig.colorbar(surf)

        plt.show()

def constructInnerFunctionplotting(B):
    if draw == 1 or draw == 2:
        print("Displaying final town structure...")
        plottowns(B.leafTowns)

    if plotfxn:
        print("Plotting resulting function ψ...")
        fig2 = plt.figure(figsize=(12, 6))
        plotfxn_func(B.leafTowns, 0)
        plt.title("Resulting Function ψ")
        plt.xlabel("x")
        plt.ylabel("ψ(x)")
        plt.grid(True)
        plt.show()

    if plotPhi:
        print("Creating Phi surface plots...")
        plotPhi_func(B)

def constructInnerFunction():
    print("=" * 70)
    print("LIPSCHITZ ALGORITHM")
    print("=" * 70)
    print(f"MPFR Precision: {gmpy2.get_context().precision} bits")
    print(f"Matrix Solver: NumPy float64")
    print(f"Dimension: {n}")
    print(f"Iterations: {J}")
    print(f"Draw mode: {draw} (0=none, 1=all iterations, 2=final only)")
    print(f"Plot function: {plotfxn}")
    print(f"Plot Phi: {plotPhi}")
    print("=" * 70)

    B = fullRefine(J)

    constructInnerFunctionplotting(B)

    if draw == 1 or draw == 2:
        print("Displaying binary tree visualization...")
        visualizeBT(B)

    print("\nAlgorithm completed successfully")
    return B

if __name__ == "__main__":
    constructInnerFunction()
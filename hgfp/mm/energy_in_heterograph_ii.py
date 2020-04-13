""" Calculate energy within a graph.

"""
# =============================================================================
# imports
# =============================================================================
import torch
import dgl
import hgfp

# =============================================================================
# module functions
# =============================================================================
def reduce_stack(msg, out):
    def _reduce_stack(nodes, msg=msg, out=out):
        return {out: nodes.mailbox[msg]}
    return _reduce_stack

def u(
        g,
        one_four_scaling=0.5,
        switch=1.0,
        damping=1e-3):
    """ Calculate energy based on two graphs, one with geometry, one with
    parameters.

    """
    scaling = {'one_four': one_four_scaling, 'nonbonded': 1.0}


    # pass the geometry onto higher-order nodes
    g.multi_update_all(
        {'bond_as_%s_in_angle' % idx:(
            dgl.function.copy_src(src='x', out='m_x_%s' % idx),
            reduce_stack(msg='m_x_%s' % idx, out='x_%s_bond' % idx)) for idx in range(2)},
        'sum')

    g.multi_update_all(
        {'bond_as_%s_in_torsion' % idx:(
            dgl.function.copy_src(src='x', out='m_x_%s' % idx),
            reduce_stack(msg='m_x_%s' % idx, out='x_%s_bond' % idx)) for idx in range(3)},
        'sum')

    g.multi_update_all(
        {'angle_as_%s_in_torsion' % idx:(
            dgl.function.copy_src(src='x', out='m_x_%s' % idx),
            reduce_stack(msg='m_x_%s' % idx, out='x_%s_angle' % idx)) for idx in range(2)},
        'sum')

    # pass the parameters onto higher-order nodes
    g.multi_update_all(
        {'bond_as_%s_in_angle' % idx:(
            dgl.function.copy_src(src='eq', out='m_eq_%s' % idx),
            reduce_stack(msg='m_eq_%s' % idx, out='eq_%s_bond' % idx)) for idx in range(2)},
        'sum')

    g.multi_update_all(
        {'bond_as_%s_in_torsion' % idx:(
            dgl.function.copy_src(src='eq', out='m_eq_%s' % idx),
            reduce_stack(msg='m_eq_%s' % idx, out='eq_%s_bond' % idx)) for idx in range(3)},
        'sum')

    g.multi_update_all(
        {'angle_as_%s_in_torsion' % idx:(
            dgl.function.copy_src(src='eq', out='m_eq_%s' % idx),
            reduce_stack(msg='m_eq_%s' % idx, out='eq_%s_angle' % idx)) for idx in range(2)},
        'sum')


    # ~~~~ 
    # bond
    # ~~~~
    # polynomial
    g.nodes['bond'].data['energy'] = hgfp.mm.energy_ii.bond(
        g.nodes['bond'].data['x'],
        g.nodes['bond'].data['k'],
        g.nodes['bond'].data['eq'])

    # ~~~~~
    # angle
    # ~~~~~
    # polynomial
    g.nodes['angle'].data['energy'] = hgfp.mm.energy_ii.angle(
        g.nodes['angle'].data['x'],
        g.nodes['angle'].data['k'],
        g.nodes['angle'].data['eq'])


    # bond-bond
    g.nodes['angle'].data['energy'] += hgfp.mm.energy_ii.bond_bond(
        g.nodes['angle'].data['x_0_bond'][0],
        g.nodes['angle'].data['x_1_bond'][0],
        g.nodes['angle'].data['k_bond_bond'],
        g.nodes['angle'].data['eq_0_bond'][0],
        g.nodes['angle'].data['eq_1_bond'][0])

    # angle-bond
    for idx in range(2):
        g.nodes['angle'].data['energy'] += hgfp.mm.energy_ii.bond_angle(
            g.nodes['angle'].data['x'],
            g.nodes['angle'].data['x_%s_bond' % idx][0],
            g.nodes['angle'].data['k_angle_bond'],
            g.nodes['angle'].data['eq'],
            g.nodes['angle'].data['eq_%s_bond' % idx])[0]


    # ~~~~~~~
    # torsion
    # ~~~~~~~
    # polynomial
    g.nodes['torsion'].data['energy'] = hgfp.mm.energy_ii.torsion(
        g.nodes['torsion'].data['x'],
        g.nodes['torsion'].data['k'])

    # torsion bond
    for idx in [0, 2]:
        g.nodes['torsion'].data['energy'] += hgfp.mm.energy_ii.torsion_bond(
            g.nodes['torsion'].data['x'],
            g.nodes['torsion'].data['x_%s_bond' % idx][0],
            g.nodes['torsion'].data['k_torsion_bond_side'],
            g.nodes['torsion'].data['eq_%s_bond' % idx][0])

    for idx in [1]:
        g.nodes['torsion'].data['energy'] += hgfp.mm.energy_ii.torsion_bond(
            g.nodes['torsion'].data['x'],
            g.nodes['torsion'].data['x_%s_bond' % idx][0],
            g.nodes['torsion'].data['k_torsion_bond_center'],
            g.nodes['torsion'].data['eq_%s_bond' % idx][0])


    # torsion angle
    for idx in range(2):
        g.nodes['torsion'].data['energy'] += hgfp.mm.energy_ii.torsion_angle(
            g.nodes['torsion'].data['x'],
            g.nodes['torsion'].data['x_%s_angle' % idx][0],
            g.nodes['torsion'].data['k_torsion_angle'],
            g.nodes['torsion'].data['eq_%s_angle' % idx][0])

    for term in ['one_four', 'nonbonded']:
        if 'x' in g.nodes[term].data and 'sigma_pair' in g.nodes[term].data:
            x = g.nodes[term].data['x']
            sigma_pair = g.nodes[term].data['sigma_pair']
            epsilon_pair = g.nodes[term].data['epsilon_pair']
            q_pair = g.nodes[term].data['q_pair']
            u = scaling[term] * hgfp.mm.energy_ii.lj(x, sigma_pair, epsilon_pair, switch=switch, damping=damping) +\
                hgfp.mm.energy_ii.coulomb(x, q_pair)

            g.nodes[term].data['energy'] = u
        else:
            g.nodes[term].data['energy'] = torch.zeros((g.number_of_nodes(ntype=term)))


    g.multi_update_all(
        {

            '%s_in_mol' % term: (
                dgl.function.copy_src(src='energy', out='m'),
                dgl.function.sum(msg='m', out='u' + term)) for term in [
                        'bond',
                        'angle',
                        'torsion',
                        'one_four',
                        'nonbonded'
                    ]
        },
        'stack')

    return g

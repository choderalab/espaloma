""" Calculate the geometric values of graph entites.

"""
# =============================================================================
# imports
# =============================================================================
import torch
import dgl

# =============================================================================
# utility functions
# =============================================================================
def reduce_stack(msg, out):
    def _reduce_stack(nodes, msg=msg, out=out):
        return {out: nodes.mailbox[msg]}
    return _reduce_stack

def distance(nodes):
    xyz_cat = nodes.data['xyz_cat']
    return {'x': torch.norm(xyz_cat[:, 0, :] - xyz_cat[:, 1, :], p=2, dim=-1)}

def angle_vl(nodes):
    # get the nodes
    xyz_center = nodes.data['xyz_center']
    xyz_side = nodes.data['xyz_side']

    # get the two vectors composing the angle
    d_xyz_side_to_center = xyz_center - xyz_side
    left = d_xyz_side_to_center[:, 0, :]
    right = d_xyz_side_to_center[:, 1, :]

    # calculate the value of the angle
    _angle_vl = torch.atan2(
        torch.norm(
            torch.cross(
                left,
                right),
            dim=-1),
        torch.sum(
            torch.mul(
                left,
                right),
            dim=-1))

    return {'x': _angle_vl}

def torsion_vl(nodes):
    # get the nodes
    xyz = torch.cat(
        (
            nodes.data['xyz0'],
            nodes.data['xyz1'],
            nodes.data['xyz2'],
            nodes.data['xyz3'],
        ),
        axis=1)

    left = torch.cross(
        xyz[:, 1, :] - xyz[:, 0, :],
        xyz[:, 1, :] - xyz[:, 2, :])

    right = torch.cross(
        xyz[:, 2, :] - xyz[:, 3, :],
        xyz[:, 2, :] - xyz[:, 1, :])

    _torsion_vl = torch.atan2(
        torch.norm(
            torch.cross(
                left,
                right),
            dim=-1),
        torch.sum(
            torch.mul(
                left,
                right),
            dim=-1))

    return {'x': _torsion_vl}

def from_heterograph_with_xyz(g):
    g.multi_update_all(
        {
            'atom_in_bond':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz_cat')),

            'atom_as_center_in_angle':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz_center')),
            'atom_as_side_in_angle':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz_side')),

            'atom_as_0_in_torsion':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz0')),
            'atom_as_1_in_torsion':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz1')),
            'atom_as_2_in_torsion':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz2')),
            'atom_as_3_in_torsion':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz3')),

            'atom_in_one_four':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz_cat')),

            'atom_in_nonbonded':(
                dgl.function.copy_src(src='xyz', out='m'),
                reduce_stack(msg='m', out='xyz_cat')),

        },
        'stack')

    g.apply_nodes(distance, ntype='bond')
    g.apply_nodes(angle_vl, ntype='angle')
    g.apply_nodes(torsion_vl, ntype='torsion')
    g.apply_nodes(distance, ntype='one_four')
    g.apply_nodes(distance, ntype='nonbonded')

    return g

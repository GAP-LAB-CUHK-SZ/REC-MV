def visualize_curve(curve)

        cano_nx = self.cano_nx.detach().cpu()[:,0,:]
        N = curves.shape[0]

        diff_a = curves[:, :-1, :] - curves[:, 1:, :]
        diff_b = curves[:,-1:,:] - curves[:,0:1, :]
        diff_a =  torch.cat([diff_a, diff_b], dim = 1)
        diff_a = diff_a / (diff_a.norm(dim = -1, keepdim = True) + 1e-6)
        cano_nx = cano_nx[:, None, :].expand_as(diff_a)


        # rotate
        cross_n = torch.cross(diff_a, cano_nx)

        dot_n = diff_a * (diff_a * cano_nx)

        rotate_nx_list = []
        for i in range(0, 360, 360 // num_joints):
            radius = torch.tensor(np.radians(i)).float()
            rotate_nx = cano_nx * torch.cos(radius)  + cross_n * torch.sin(radius) + dot_n*(1-torch.cos(radius))
            rotate_nx_list.append(rotate_nx[..., None, :])

        rotate_nx = torch.cat(rotate_nx_list, dim=-2)

        curve_mesh_verts = curves[..., None,:] + curve_radius * rotate_nx

        face_idx = torch.arange(curve_mesh_verts.view(-1,3).shape[0]).view(*curve_mesh_verts.shape[:-1])
        # adj face
        batch_faces = []
        for f_i in range(curve_mesh_verts.shape[1] ):
            start_face = f_i % curve_mesh_verts.shape[1]
            end_face = (f_i+1) % curve_mesh_verts.shape[1]

            start_face = face_idx[:,start_face]
            end_face = face_idx[:,end_face]

            face_list = []
            for v_i in range(start_face.shape[1]):
                start_v_i  = (v_i) % start_face.shape[1]
                end_v_i = (v_i +1) % start_face.shape[1]
                face_a = torch.cat([start_face[:,start_v_i:start_v_i+1], end_face[:, start_v_i:start_v_i+1], end_face[:, end_v_i:end_v_i+1]], dim = -1)
                face_b = torch.cat([start_face[:,start_v_i:start_v_i+1], end_face[:, end_v_i:end_v_i+1],start_face[:, end_v_i:end_v_i+1]],  dim = -1)
                face_list.append(face_a)
                face_list.append(face_b)

            faces = torch.cat(face_list, dim =-1).view(N, num_joints *2,3)
            batch_faces.append(faces)

        curve_faces = torch.cat(batch_faces, dim =1)


        for curve_idx, curve_face in enumerate(curve_faces):
            curve_faces[curve_idx] -= curve_face.min()

        curve_verts = curve_mesh_verts.view(N,-1,3)
        curve_meshes = [Meshes([curve_vert],[curve_face]) for curve_vert, curve_face in zip(curve_verts, curve_faces)]

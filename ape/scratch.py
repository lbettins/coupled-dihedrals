def get_displaced_geometries_dict(self, vector, step_size, limit=15, torsion_ind=None):
        if torsion_ind is None:
            internal = get_RedundantCoords(self.symbols, self.cart_coords)
            magnitude = np.linalg.norm(vector)
            normalizes_vector = vector/magnitude
            qj = np.matmul(internal.B, normalizes_vector)
            x_dict = {0:self.cart_coords}
            positive_samples = range(limit)
            internal = get_RedundantCoords(self.symbols, self.cart_coords)
            for sample in positive_samples:
                x_dict[sample+1] = x_dict[sample] + internal.transform_int_step((qj*step_size).reshape(-1,))
            negative_samples = list(range(-limit+1, 1))
            negative_samples.reverse()
            internal = get_RedundantCoords(self.symbols, self.cart_coords)
            for sample in negative_samples:
                x_dict[sample-1] = x_dict[sample] + internal.transform_int_step((-qj*step_size).reshape(-1,))
        else:
            rotors_dict = self.rotors_dict
            internal = get_RedundantCoords(self.symbols, self.cart_coords, rotors_dict)
            B = internal.B
            Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)
            nrow = B.shape[0]
            qk = np.zeros(nrow, dtype=int)
            qk[torsion_ind] = 1
            x_dict = {0:self.cart_coords}
            limit = int(2*np.pi/step_size)
            positive_samples = range(limit)
            for sample in positive_samples:
                x_dict[sample+1] = x_dict[sample] + internal.transform_int_step((qk*step_size).reshape(-1,))
        displaced_geometries_dict = {}
        for xi in x_dict:
            xyz = getXYZ(self.symbols, x_dict[xi])
            displaced_geometries_dict[xi] = xyz
        return displaced_geometries_dict

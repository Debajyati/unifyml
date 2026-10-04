mod tests {
use unifyml::Vector;
    // ===================== Construction =====================

    #[test]
    fn new_creates_empty_vector() {
        let v: Vector<f32> = Vector::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn default_creates_empty_vector() {
        let v: Vector<f32> = Vector::default();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn zeros_creates_correct_length_and_values() {
        let v = Vector::<f32>::zeros(5);
        assert_eq!(v.len(), 5);
        for i in 0..5 {
            assert_eq!(v[i], 0.0);
        }
    }

    #[test]
    fn zeros_with_zero_length() {
        let v = Vector::<f32>::zeros(0);
        assert!(v.is_empty());
    }

    #[test]
    fn ones_creates_correct_length_and_values() {
        let v = Vector::<f32>::ones(4);
        assert_eq!(v.len(), 4);
        for i in 0..4 {
            assert_eq!(v[i], 1.0);
        }
    }

    #[test]
    fn ones_with_zero_length() {
        let v = Vector::<f32>::ones(0);
        assert!(v.is_empty());
    }

    #[test]
    fn from_vec_stores_elements_correctly() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn from_vec_empty() {
        let v: Vector<f32> = Vector::from_vec(vec![]);
        assert!(v.is_empty());
    }

    #[test]
    fn from_slice_stores_elements_correctly() {
        let arr = [4.0, 5.0, 6.0];
        let v = Vector::from_slice(&arr);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 4.0);
        assert_eq!(v[1], 5.0);
        assert_eq!(v[2], 6.0);
    }

    #[test]
    fn from_slice_is_independent_copy() {
        let mut arr = [1.0, 2.0, 3.0];
        let v = Vector::from_slice(&arr);
        arr[0] = 99.0;
        assert_eq!(v[0], 1.0, "Vector should own its data, not alias the slice");
    }

    #[test]
    fn from_slice_empty() {
        let arr: [f32; 0] = [];
        let v = Vector::from_slice(&arr);
        assert!(v.is_empty());
    }

    // ===================== Access =====================

    #[test]
    fn get_returns_some_for_valid_index() {
        let v = Vector::from_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!(v.get(0), Some(&10.0));
        assert_eq!(v.get(1), Some(&20.0));
        assert_eq!(v.get(2), Some(&30.0));
    }

    #[test]
    fn get_returns_none_for_out_of_bounds() {
        let v = Vector::from_vec(vec![10.0, 20.0]);
        assert_eq!(v.get(2), None);
        assert_eq!(v.get(100), None);
    }

    #[test]
    fn get_on_empty_vector_returns_none() {
        let v: Vector<f32> = Vector::new();
        assert_eq!(v.get(0), None);
    }

    #[test]
    fn index_operator_reads_correctly() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    #[should_panic]
    fn index_operator_panics_on_out_of_bounds() {
        let v = Vector::from_vec(vec![1.0, 2.0]);
        let _ = v[5];
    }

    #[test]
    fn index_mut_allows_mutation() {
        let mut v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        v[1] = 99.0;
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 99.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    #[should_panic]
    fn index_mut_panics_on_out_of_bounds() {
        let mut v = Vector::from_vec(vec![1.0]);
        v[10] = 0.0;
    }

    #[test]
    fn as_slice_returns_correct_view() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn as_mut_slice_allows_in_place_mutation() {
        let mut v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        v.as_mut_slice()[0] = 42.0;
        assert_eq!(v[0], 42.0);
    }

    // ===================== Operations =====================

    #[test]
    fn add_two_vectors() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c, Vector::from_vec(vec![5.0, 7.0, 9.0]));
    }

    #[test]
    fn sub_two_vectors() {
        let a = Vector::from_vec(vec![10.0, 20.0, 30.0]);
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c, Vector::from_vec(vec![9.0, 18.0, 27.0]));
    }

    #[test]
    fn sub_produces_negative_values() {
        let a = Vector::from_vec(vec![1.0, 2.0]);
        let b = Vector::from_vec(vec![5.0, 10.0]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c, Vector::from_vec(vec![-4.0, -8.0]));
    }

    #[test]
    fn scale_by_positive_factor() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = v.scale(3.0);
        assert_eq!(scaled, Vector::from_vec(vec![3.0, 6.0, 9.0]));
    }

    #[test]
    fn scale_by_zero() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = v.scale(0.0);
        assert_eq!(scaled, Vector::zeros(3));
    }

    #[test]
    fn scale_by_negative_factor() {
        let v = Vector::from_vec(vec![1.0, -2.0, 3.0]);
        let scaled = v.scale(-2.0);
        assert_eq!(scaled, Vector::from_vec(vec![-2.0, 4.0, -6.0]));
    }

    #[test]
    fn scale_does_not_mutate_original() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let _ = v.scale(10.0);
        assert_eq!(v, Vector::from_vec(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn dot_product_basic() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(a.dot(&b).unwrap(), 32.0);
    }

    #[test]
    fn dot_product_with_zeros() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::<f32>::zeros(3);
        assert_eq!(a.dot(&b).unwrap(), 0.0);
    }

    #[test]
    fn dot_product_is_commutative() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b).unwrap(), b.dot(&a).unwrap());
    }

    #[test]
    fn sum_of_elements() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn sum_with_negative_values() {
        let v = Vector::from_vec(vec![10.0, -3.0, -7.0]);
        assert_eq!(v.sum(), 0.0);
    }

    // ===================== Edge Cases =====================

    // --- Dimension mismatch ---

    #[test]
    fn add_dimension_mismatch_returns_err() {
        let a = Vector::from_vec(vec![1.0, 2.0]);
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn sub_dimension_mismatch_returns_err() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![1.0]);
        assert!(a.sub(&b).is_err());
    }

    #[test]
    fn dot_dimension_mismatch_returns_err() {
        let a = Vector::from_vec(vec![1.0, 2.0]);
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(a.dot(&b).is_err());
    }

    // --- Empty vectors ---

    #[test]
    fn add_empty_vectors() {
        let a: Vector<f32> = Vector::new();
        let b: Vector<f32> = Vector::new();
        let c = a.add(&b).unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn sub_empty_vectors() {
        let a: Vector<f32> = Vector::new();
        let b: Vector<f32> = Vector::new();
        let c = a.sub(&b).unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn dot_empty_vectors_returns_zero() {
        let a: Vector<f32> = Vector::new();
        let b: Vector<f32> = Vector::new();
        assert_eq!(a.dot(&b).unwrap(), 0.0);
    }

    #[test]
    fn sum_empty_vector_returns_zero() {
        let v: Vector<f32> = Vector::new();
        assert_eq!(v.sum(), 0.0);
    }

    #[test]
    fn scale_empty_vector() {
        let v: Vector<f32> = Vector::new();
        let scaled = v.scale(5.0);
        assert!(scaled.is_empty());
    }

    // --- Single-element vectors ---

    #[test]
    fn add_single_element() {
        let a = Vector::from_vec(vec![3.0]);
        let b = Vector::from_vec(vec![7.0]);
        assert_eq!(a.add(&b).unwrap(), Vector::from_vec(vec![10.0]));
    }

    #[test]
    fn sub_single_element() {
        let a = Vector::from_vec(vec![10.0]);
        let b = Vector::from_vec(vec![3.0]);
        assert_eq!(a.sub(&b).unwrap(), Vector::from_vec(vec![7.0]));
    }

    #[test]
    fn dot_single_element() {
        let a = Vector::from_vec(vec![4.0]);
        let b = Vector::from_vec(vec![5.0]);
        assert_eq!(a.dot(&b).unwrap(), 20.0);
    }

    #[test]
    fn sum_single_element() {
        let v = Vector::from_vec(vec![42.0]);
        assert_eq!(v.sum(), 42.0);
    }

    #[test]
    fn scale_single_element() {
        let v = Vector::from_vec(vec![5.0]);
        assert_eq!(v.scale(3.0), Vector::from_vec(vec![15.0]));
    }

    // --- Clone & PartialEq (derived traits) ---

    #[test]
    fn clone_creates_independent_copy() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut b = a.clone();
        b[0] = 99.0;
        assert_eq!(a[0], 1.0, "Original should be unaffected by clone mutation");
    }

    #[test]
    fn partial_eq_works() {
        let a = Vector::from_vec(vec![1.0, 2.0]);
        let b = Vector::from_vec(vec![1.0, 2.0]);
        let c = Vector::from_vec(vec![1.0, 3.0]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

// Unit tests for the C boundary (v0.1.3). They call the `extern "C"` functions as C would.
use super::*;
use std::ptr::{null, null_mut};

const JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 3, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}}
    ],
    "special_tokens": {
      "[CLS]": {"id": "[CLS]", "ids": [3], "tokens": ["[CLS]"]}
    }
  },
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "[CLS]": 3},
    "unk_token": "[UNK]"
  }
}"#;

fn make() -> *mut TokenizerWrapper {
    let h = tokenizers_new_from_str(JSON.as_ptr(), JSON.len());
    assert!(!h.is_null(), "valid JSON must give a handle: {}", last_error());
    h
}

fn last_error() -> String {
    let mut p: *const u8 = null();
    let mut n: usize = 0;
    tokenizers_get_last_error(&mut p, &mut n);
    if n == 0 {
        return String::new();
    }
    unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(p, n)).into_owned() }
}

fn ids_of(r: &TokenizerEncodeResult) -> Vec<u32> {
    if r.len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(r.token_ids, r.len).to_vec() }
}

fn empty() -> TokenizerEncodeResult {
    TokenizerEncodeResult {
        token_ids: null_mut(),
        len: 0,
    }
}

fn is_empty(r: &TokenizerEncodeResult) -> bool {
    r.token_ids.is_null() && r.len == 0
}

fn encode(h: *mut TokenizerWrapper, text: &[u8], special: i32) -> (i32, TokenizerEncodeResult) {
    // Pre-fill with garbage so we can see the function wrote {NULL,0} on failure.
    let mut r = TokenizerEncodeResult {
        token_ids: 0x1 as *mut u32,
        len: 77,
    };
    let st = tokenizers_encode(h, text.as_ptr(), text.len(), special, &mut r);
    (st, r)
}

// 1
#[test]
fn encode_valid() {
    let h = make();
    let (st, mut r) = encode(h, b"hello world", 0);
    assert_eq!(st, TOKENIZERS_OK);
    assert_eq!(ids_of(&r), vec![1, 2]);
    tokenizers_free_encode_results(&mut r, 1);

    let (st, mut r) = encode(h, b"hello world", 1);
    assert_eq!(st, TOKENIZERS_OK);
    assert_eq!(ids_of(&r), vec![3, 1, 2]);
    tokenizers_free_encode_results(&mut r, 1);
    tokenizers_free(h);
}

// 2
#[test]
fn bad_json_gives_null() {
    let bad = b"{not json";
    let h = tokenizers_new_from_str(bad.as_ptr(), 9);
    assert!(h.is_null());
    assert!(!last_error().is_empty());

    let h = tokenizers_new_from_str(null(), 0);
    assert!(h.is_null());
    assert!(!last_error().is_empty());
    // freeing NULL is a no-op
    tokenizers_free(null_mut());
}

// 3
#[test]
fn encode_invalid_utf8() {
    let h = make();
    let (st, r) = encode(h, &[0x68, 0xFF, 0x69], 0);
    assert_eq!(st, TOKENIZERS_ERR_INVALID_UTF8);
    assert!(is_empty(&r));
    assert!(!last_error().is_empty());
    tokenizers_free(h);
}

// 4
#[test]
fn encode_null_args() {
    let h = make();
    let mut r = empty();
    assert_eq!(tokenizers_encode(h, null(), 0, 0, &mut r), TOKENIZERS_OK);
    // An empty encoding is {NULL, 0}: nothing to free.
    assert!(r.token_ids.is_null());
    assert_eq!(r.len, 0);
    tokenizers_free_encode_results(&mut r, 1);

    let mut r = TokenizerEncodeResult {
        token_ids: 0x1 as *mut u32,
        len: 9,
    };
    assert_eq!(tokenizers_encode(h, null(), 5, 0, &mut r), TOKENIZERS_ERR_NULL_ARG);
    assert!(is_empty(&r));

    let text = b"hello";
    let mut r = TokenizerEncodeResult {
        token_ids: 0x1 as *mut u32,
        len: 9,
    };
    assert_eq!(
        tokenizers_encode(null_mut(), text.as_ptr(), text.len(), 0, &mut r),
        TOKENIZERS_ERR_NULL_ARG
    );
    assert!(is_empty(&r));

    assert_eq!(
        tokenizers_encode(h, text.as_ptr(), text.len(), 0, null_mut()),
        TOKENIZERS_ERR_NULL_ARG
    );
    tokenizers_free(h);
}

// 5
#[test]
fn encode_batch() {
    let h = make();
    let items: [&[u8]; 3] = [b"hello", b"world hello", b""];
    let ptrs: Vec<*const u8> = items.iter().map(|s| s.as_ptr()).collect();
    let lens: Vec<usize> = items.iter().map(|s| s.len()).collect();
    let mut res = [empty(), empty(), empty()];
    let st = tokenizers_encode_batch(h, ptrs.as_ptr(), lens.as_ptr(), 3, 0, res.as_mut_ptr());
    assert_eq!(st, TOKENIZERS_OK, "{}", last_error());
    assert_eq!(ids_of(&res[0]), vec![1]);
    assert_eq!(ids_of(&res[1]), vec![2, 1]);
    assert_eq!(ids_of(&res[2]), Vec::<u32>::new());
    assert!(res[2].token_ids.is_null(), "empty batch item must be (NULL, 0)");
    tokenizers_free_encode_results(res.as_mut_ptr(), 3);

    let bad: [&[u8]; 3] = [b"hello", &[0x68, 0xFF, 0x69], b"world"];
    let ptrs: Vec<*const u8> = bad.iter().map(|s| s.as_ptr()).collect();
    let lens: Vec<usize> = bad.iter().map(|s| s.len()).collect();
    let mut res = [
        TokenizerEncodeResult { token_ids: 0x1 as *mut u32, len: 5 },
        TokenizerEncodeResult { token_ids: 0x1 as *mut u32, len: 5 },
        TokenizerEncodeResult { token_ids: 0x1 as *mut u32, len: 5 },
    ];
    let st = tokenizers_encode_batch(h, ptrs.as_ptr(), lens.as_ptr(), 3, 0, res.as_mut_ptr());
    assert_eq!(st, TOKENIZERS_ERR_INVALID_UTF8);
    assert!(res.iter().all(is_empty));
    assert!(!last_error().is_empty());

    assert_eq!(
        tokenizers_encode_batch(h, null(), null(), 0, 0, null_mut()),
        TOKENIZERS_OK
    );
    // The handle is checked before the num_seqs == 0 early return.
    assert_eq!(
        tokenizers_encode_batch(null_mut(), null(), null(), 0, 0, null_mut()),
        TOKENIZERS_ERR_NULL_ARG
    );
    assert!(last_error().contains("handle"));
    // num_seqs > 0 with NULL arrays
    let mut res = [empty()];
    assert_eq!(
        tokenizers_encode_batch(h, null(), lens.as_ptr(), 1, 0, res.as_mut_ptr()),
        TOKENIZERS_ERR_NULL_ARG
    );
    assert_eq!(
        tokenizers_encode_batch(h, ptrs.as_ptr(), null(), 1, 0, res.as_mut_ptr()),
        TOKENIZERS_ERR_NULL_ARG
    );
    assert_eq!(
        tokenizers_encode_batch(h, ptrs.as_ptr(), lens.as_ptr(), 1, 0, null_mut()),
        TOKENIZERS_ERR_NULL_ARG
    );
    tokenizers_free(h);
}

// 6
#[test]
fn free_results_null_and_double() {
    tokenizers_free_encode_results(null_mut(), 3);
    let h = make();
    let (st, mut r) = encode(h, b"hello world", 1);
    assert_eq!(st, TOKENIZERS_OK);
    tokenizers_free_encode_results(&mut r, 1);
    assert!(is_empty(&r));
    tokenizers_free_encode_results(&mut r, 1);
    assert!(is_empty(&r));
    tokenizers_free(h);
}

fn decode_str(h: *mut TokenizerWrapper) -> Vec<u8> {
    let mut p: *mut u8 = null_mut();
    let mut n: usize = 0;
    assert_eq!(tokenizers_get_decode_str(h, &mut p, &mut n), TOKENIZERS_OK);
    if n == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(p, n).to_vec() }
}

// 7
#[test]
fn decode() {
    let h = make();
    let ids = [1u32, 2];
    assert_eq!(tokenizers_decode(h, ids.as_ptr(), 2, 0), TOKENIZERS_OK);
    assert_eq!(decode_str(h), b"hello world".to_vec());

    assert_eq!(tokenizers_decode(null_mut(), ids.as_ptr(), 2, 0), TOKENIZERS_ERR_NULL_ARG);

    assert_eq!(tokenizers_decode(h, null(), 0, 0), TOKENIZERS_OK);
    assert_eq!(decode_str(h), Vec::<u8>::new());

    // NULL ids with len > 0: error, decode string emptied
    assert_eq!(tokenizers_decode(h, ids.as_ptr(), 2, 0), TOKENIZERS_OK);
    assert_eq!(tokenizers_decode(h, null(), 2, 0), TOKENIZERS_ERR_NULL_ARG);
    assert_eq!(decode_str(h), Vec::<u8>::new());

    let mut n: usize = 0;
    assert_eq!(tokenizers_get_decode_str(h, null_mut(), &mut n), TOKENIZERS_ERR_NULL_ARG);
    let mut p: *mut u8 = null_mut();
    assert_eq!(tokenizers_get_decode_str(null_mut(), &mut p, &mut n), TOKENIZERS_ERR_NULL_ARG);
    tokenizers_free(h);
}

// 8
#[test]
fn vocab_queries() {
    let h = make();
    let mut size: usize = 0;
    assert_eq!(tokenizers_get_vocab_size(h, &mut size), TOKENIZERS_OK);
    assert_eq!(size, 4);

    let mut p: *mut u8 = null_mut();
    let mut n: usize = 0;
    assert_eq!(tokenizers_id_to_token(h, 2, &mut p, &mut n), TOKENIZERS_OK);
    assert_eq!(unsafe { std::slice::from_raw_parts(p, n) }, b"world");
    assert_eq!(tokenizers_id_to_token(h, 999, &mut p, &mut n), TOKENIZERS_OK);
    assert_eq!(n, 0);

    let mut id: i32 = 0;
    let w = b"world";
    assert_eq!(tokenizers_token_to_id(h, w.as_ptr(), w.len(), &mut id), TOKENIZERS_OK);
    assert_eq!(id, 2);
    let z = b"zzz";
    assert_eq!(tokenizers_token_to_id(h, z.as_ptr(), z.len(), &mut id), TOKENIZERS_OK);
    assert_eq!(id, -1);

    assert_eq!(tokenizers_get_vocab_size(h, null_mut()), TOKENIZERS_ERR_NULL_ARG);
    assert_eq!(tokenizers_get_vocab_size(null_mut(), &mut size), TOKENIZERS_ERR_NULL_ARG);
    assert_eq!(tokenizers_id_to_token(h, 2, null_mut(), &mut n), TOKENIZERS_ERR_NULL_ARG);
    assert_eq!(tokenizers_id_to_token(h, 2, &mut p, null_mut()), TOKENIZERS_ERR_NULL_ARG);
    assert_eq!(tokenizers_id_to_token(null_mut(), 2, &mut p, &mut n), TOKENIZERS_ERR_NULL_ARG);
    assert_eq!(
        tokenizers_token_to_id(h, w.as_ptr(), w.len(), null_mut()),
        TOKENIZERS_ERR_NULL_ARG
    );
    assert_eq!(
        tokenizers_token_to_id(null_mut(), w.as_ptr(), w.len(), &mut id),
        TOKENIZERS_ERR_NULL_ARG
    );
    assert_eq!(tokenizers_token_to_id(h, null(), 3, &mut id), TOKENIZERS_ERR_NULL_ARG);
    tokenizers_free(h);
}

// 8b: ids >= 2^31 do not fit int32_t; they must be an error, not a wrap to a negative id.
#[test]
fn token_to_id_overflow() {
    let json = r#"{
  "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
  "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"[UNK]": 0, "ok": 2147483647, "big": 2147483648, "max": 4294967295},
    "unk_token": "[UNK]"
  }
}"#;
    let h = tokenizers_new_from_str(json.as_ptr(), json.len());
    assert!(!h.is_null(), "{}", last_error());
    let mut id: i32 = 7;
    let t = b"ok";
    assert_eq!(tokenizers_token_to_id(h, t.as_ptr(), t.len(), &mut id), TOKENIZERS_OK);
    assert_eq!(id, i32::MAX);
    for t in [&b"big"[..], &b"max"[..]] {
        let mut id: i32 = 7;
        assert_eq!(
            tokenizers_token_to_id(h, t.as_ptr(), t.len(), &mut id),
            TOKENIZERS_ERR_TOKENIZER
        );
        assert!(last_error().contains("int32"), "{}", last_error());
    }
    tokenizers_free(h);

    assert_eq!(id_to_i32(0).ok(), Some(0));
    assert_eq!(id_to_i32(i32::MAX as u32).ok(), Some(i32::MAX));
    assert!(id_to_i32(1u32 << 31).is_err());
    assert_eq!(id_to_i32(u32::MAX).err().map(|e| e.code), Some(TOKENIZERS_ERR_TOKENIZER));
}

fn bpe(vocab: &str, merges: &str, added: &str) -> *mut TokenizerWrapper {
    byte_level_bpe_tokenizers_new_from_str(
        vocab.as_ptr(),
        vocab.len(),
        merges.as_ptr(),
        merges.len(),
        added.as_ptr(),
        added.len(),
    )
}

// 9
#[test]
fn byte_level_bpe_errors() {
    let vocab = r#"{"a":0,"b":1,"c":2,"ab":3}"#;
    let ok = bpe(vocab, "#version: 0.2\na b\n", "");
    assert!(!ok.is_null(), "{}", last_error());
    tokenizers_free(ok);

    let h = bpe(vocab, "a b c", "");
    assert!(h.is_null());
    let msg = last_error();
    assert!(msg.contains("merges"), "{}", msg);

    let h = bpe(vocab, "a b", "[1,2]");
    assert!(h.is_null());
    assert!(last_error().contains("added_tokens"));

    let h = bpe("{not json", "a b", "");
    assert!(h.is_null());
    let h = bpe("[1]", "a b", "");
    assert!(h.is_null());
    // merge referencing tokens not in vocab: BPE builder error, not a panic
    let h = bpe(vocab, "x y", "");
    assert!(h.is_null());
    assert!(!last_error().is_empty());
}

extern "C" fn force_panic_status() -> i32 {
    guard_status(|| panic!("forced test panic"))
}

extern "C" fn force_panic_ptr() -> *mut TokenizerWrapper {
    guard_ptr(|| panic!("{}", String::from("forced ptr panic")))
}

// 10
#[test]
fn panic_is_contained() {
    assert_eq!(force_panic_status(), TOKENIZERS_ERR_PANIC);
    assert!(last_error().contains("forced test panic"));
    assert!(force_panic_ptr().is_null());
    assert!(last_error().contains("forced ptr panic"));
}

// 11
#[test]
fn last_error_is_thread_local() {
    let a = std::thread::spawn(|| {
        let h = tokenizers_new_from_str(b"{not json".as_ptr(), 9);
        assert!(h.is_null());
        last_error().len()
    })
    .join()
    .expect("thread A");
    assert!(a > 0);
    let b = std::thread::spawn(|| {
        let mut p: *const u8 = null();
        let mut n: usize = 123;
        tokenizers_get_last_error(&mut p, &mut n);
        // null out-pointers are ignored
        tokenizers_get_last_error(null_mut(), null_mut());
        n
    })
    .join()
    .expect("thread B");
    assert_eq!(b, 0);
}

#[test]
fn success_leaves_last_error_unchanged() {
    let h = tokenizers_new_from_str(b"{not json".as_ptr(), 9);
    assert!(h.is_null());
    let before = last_error();
    let h = make();
    let (st, mut r) = encode(h, b"hello", 0);
    assert_eq!(st, TOKENIZERS_OK);
    tokenizers_free_encode_results(&mut r, 1);
    assert_eq!(last_error(), before);
    tokenizers_free(h);
}

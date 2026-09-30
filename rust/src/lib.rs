// A simple C wrapper of tokenzier library
//
// v0.1.3: every C entry point returns a status code (or NULL for constructors) instead of
// panicking across `extern "C"`. Panics are caught at the boundary with `catch_unwind`.
// The message of the most recent failure on the calling thread is available through
// `tokenizers_get_last_error`.
//
// v0.1.4: adds `tokenizers_encode_batch_truncated` (HF truncation applied inside Rust, before
// the post-processor adds special tokens). Purely additive.
use ahash::AHashMap;
use serde_json::Value;
use std::cell::RefCell;
use std::convert::TryFrom;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::str::FromStr;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{
    PostProcessor, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

pub const TOKENIZERS_OK: i32 = 0;
pub const TOKENIZERS_ERR_NULL_ARG: i32 = -1;
pub const TOKENIZERS_ERR_INVALID_UTF8: i32 = -2;
pub const TOKENIZERS_ERR_TOKENIZER: i32 = -3;
pub const TOKENIZERS_ERR_PANIC: i32 = -4;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    decode_str: String,
    id_to_token_result: String,
}

pub type Vocab = AHashMap<String, u32>;
pub type Merges = Vec<(String, String)>;

#[repr(C)]
pub struct TokenizerEncodeResult {
    token_ids: *mut u32,
    len: usize,
}

/// A failure carried back to the C boundary: status code plus message.
struct CError {
    code: i32,
    msg: String,
}

impl CError {
    fn new(code: i32, msg: impl Into<String>) -> CError {
        CError {
            code,
            msg: msg.into(),
        }
    }
}

type CResult<T> = Result<T, CError>;

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

fn set_last_error(msg: &str) {
    // try_with: never panic, even during thread teardown.
    let _ = LAST_ERROR.try_with(|e| {
        if let Ok(mut s) = e.try_borrow_mut() {
            s.clear();
            s.push_str(msg);
        }
    });
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        String::from("unknown panic")
    }
}

/// Runs `f` with panics caught. Returns the status code; sets the last error on failure.
fn guard_status<F: FnOnce() -> CResult<()>>(f: F) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => TOKENIZERS_OK,
        Ok(Err(e)) => {
            set_last_error(&e.msg);
            e.code
        }
        Err(payload) => {
            set_last_error(&format!("panic: {}", panic_message(&*payload)));
            TOKENIZERS_ERR_PANIC
        }
    }
}

/// Runs a constructor with panics caught. Returns NULL on any failure.
fn guard_ptr<F: FnOnce() -> CResult<TokenizerWrapper>>(f: F) -> *mut TokenizerWrapper {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(w)) => Box::into_raw(Box::new(w)),
        Ok(Err(e)) => {
            set_last_error(&e.msg);
            std::ptr::null_mut()
        }
        Err(payload) => {
            set_last_error(&format!("panic: {}", panic_message(&*payload)));
            std::ptr::null_mut()
        }
    }
}

/// `(NULL, 0)` is the empty slice; `(NULL, n > 0)` is ERR_NULL_ARG. Never calls
/// `from_raw_parts` with a null pointer.
unsafe fn bytes<'a>(p: *const u8, n: usize, what: &str) -> CResult<&'a [u8]> {
    if n == 0 {
        Ok(&[])
    } else if p.is_null() {
        Err(CError::new(
            TOKENIZERS_ERR_NULL_ARG,
            format!("{} is NULL with length {}", what, n),
        ))
    } else {
        Ok(std::slice::from_raw_parts(p, n))
    }
}

unsafe fn strict_str<'a>(p: *const u8, n: usize, what: &str) -> CResult<&'a str> {
    let b = bytes(p, n, what)?;
    std::str::from_utf8(b).map_err(|e| {
        CError::new(
            TOKENIZERS_ERR_INVALID_UTF8,
            format!("{} is not valid UTF-8: {}", what, e),
        )
    })
}

unsafe fn handle_mut<'a>(h: *mut TokenizerWrapper) -> CResult<&'a mut TokenizerWrapper> {
    h.as_mut()
        .ok_or_else(|| CError::new(TOKENIZERS_ERR_NULL_ARG, "tokenizer handle is NULL"))
}

fn null_arg(what: &str) -> CError {
    CError::new(TOKENIZERS_ERR_NULL_ARG, format!("{} is NULL", what))
}

fn tok_err(context: &str, e: impl std::fmt::Display) -> CError {
    CError::new(TOKENIZERS_ERR_TOKENIZER, format!("{}: {}", context, e))
}

fn read_id_map(json: &str, what: &str, vocab: &mut Vocab) -> CResult<()> {
    let parsed: Value = serde_json::from_str(json)
        .map_err(|e| tok_err(&format!("Invalid {} (JSON)", what), e))?;
    match parsed {
        Value::Object(m) => {
            for (token, id) in m {
                if let Value::Number(id) = id {
                    let id = id
                        .as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| {
                            CError::new(
                                TOKENIZERS_ERR_TOKENIZER,
                                format!("Invalid {}: id of '{}' is not a u32", what, token),
                            )
                        })?;
                    vocab.insert(token, id);
                }
            }
            Ok(())
        }
        _ => Err(CError::new(
            TOKENIZERS_ERR_TOKENIZER,
            format!("Invalid {}: expected a JSON object", what),
        )),
    }
}

impl TokenizerWrapper {
    fn from_str(json: &str) -> CResult<TokenizerWrapper> {
        let tokenizer = Tokenizer::from_str(json).map_err(|e| tok_err("Invalid tokenizer JSON", e))?;
        Ok(TokenizerWrapper {
            tokenizer,
            decode_str: String::new(),
            id_to_token_result: String::new(),
        })
    }

    fn byte_level_bpe_from_str(
        vocab: &str,
        merges: &str,
        added_tokens: &str,
    ) -> CResult<TokenizerWrapper> {
        let mut vocab_map: Vocab = AHashMap::new();
        read_id_map(vocab, "vocab.json file", &mut vocab_map)?;
        if !added_tokens.is_empty() {
            read_id_map(added_tokens, "added_tokens.json file", &mut vocab_map)?;
        }

        let mut merge_list: Merges = Vec::new();
        for (line_no, line) in merges.lines().enumerate() {
            if line.starts_with("#version") {
                continue;
            }
            let mut parts = line.split(' ');
            match (parts.next(), parts.next(), parts.next()) {
                (Some(a), Some(b), None) => merge_list.push((a.to_string(), b.to_string())),
                _ => {
                    return Err(CError::new(
                        TOKENIZERS_ERR_TOKENIZER,
                        format!(
                            "Invalid merges.txt file: line {} does not have exactly 2 parts",
                            line_no + 1
                        ),
                    ))
                }
            }
        }

        let bpe = BPE::builder()
            .vocab_and_merges(vocab_map, merge_list)
            .build()
            .map_err(|e| tok_err("Invalid BPE vocab/merges", e))?;
        let byte_level = ByteLevel::new(
            /*add_prefix_space=*/ false, /*trim_offsets=*/ false,
            /*use_regex=*/ false,
        );
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer
            .with_pre_tokenizer(Some(byte_level))
            .with_decoder(Some(byte_level));
        Ok(TokenizerWrapper {
            tokenizer,
            decode_str: String::new(),
            id_to_token_result: String::new(),
        })
    }

    fn encode(&mut self, text: &str, add_special_tokens: bool) -> CResult<Vec<u32>> {
        let encoded = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| tok_err("encode failed", e))?;
        Ok(encoded.get_ids().to_vec())
    }

    fn encode_batch(
        &mut self,
        texts: Vec<&str>,
        add_special_tokens: bool,
    ) -> CResult<Vec<Vec<u32>>> {
        let encoded = self
            .tokenizer
            .encode_batch(texts, add_special_tokens)
            .map_err(|e| tok_err("encode_batch failed", e))?;
        Ok(encoded
            .into_iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect())
    }

    /// Batch encode with HF truncation to `max_length` tokens (special tokens included).
    /// The tokenizer's own truncation config is restored on every path (a drop guard also
    /// covers unwinding).
    fn encode_batch_truncated(
        &mut self,
        texts: Vec<&str>,
        add_special_tokens: bool,
        max_length: usize,
    ) -> CResult<Vec<Vec<u32>>> {
        let n_added = self
            .tokenizer
            .get_post_processor()
            .map_or(0, |pp| pp.added_tokens(false));
        // tokenizers 0.21.4 computes `max_length - n_added` in unchecked usize arithmetic, in
        // `with_truncation` and in `post_process` (when special tokens are added). In a release
        // build that wraps to a huge limit, so the item comes back untruncated; Python
        // tokenizers 0.22.2 returns the same. With overflow checks on (debug/test builds) it
        // would panic instead. Produce the release/Python result explicitly so builds agree.
        let params = if add_special_tokens && max_length < n_added {
            None
        } else {
            Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            })
        };
        let saved = self.tokenizer.get_truncation().cloned();
        let mut guard = TruncationRestore {
            tokenizer: &mut self.tokenizer,
            saved: Some(saved),
        };
        set_truncation(guard.tokenizer, params)?;
        let encoded = guard
            .tokenizer
            .encode_batch(texts, add_special_tokens)
            .map_err(|e| tok_err("encode_batch failed", e))?;
        guard.restore()?;
        Ok(encoded
            .into_iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect())
    }

    fn decode(&mut self, ids: &[u32], skip_special_tokens: bool) -> CResult<()> {
        self.decode_str.clear();
        let s = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| tok_err("decode failed", e))?;
        self.decode_str = s;
        Ok(())
    }
}

/// Sets the tokenizer's truncation without running the crate's `with_truncation` check on
/// `params` itself: that check computes `max_length - n_added` unchecked (it overflows for a
/// small `max_length`) and only rejects a `stride` too large for the effective length, which
/// cannot happen with our stride 0. `with_truncation` is still called (with a seed that always
/// validates) and its `Err` maps to ERR_TOKENIZER. Also used to restore the saved config
/// verbatim, whatever tokenizer.json put there.
fn set_truncation(tokenizer: &mut Tokenizer, params: Option<TruncationParams>) -> CResult<()> {
    let params = match params {
        None => {
            tokenizer
                .with_truncation(None)
                .map_err(|e| tok_err("with_truncation failed", e))?;
            return Ok(());
        }
        Some(p) => p,
    };
    tokenizer
        .with_truncation(Some(TruncationParams {
            direction: params.direction,
            max_length: usize::MAX,
            strategy: params.strategy,
            stride: 0,
        }))
        .map_err(|e| tok_err("with_truncation failed", e))?;
    match tokenizer.get_truncation_mut() {
        Some(t) => {
            *t = params;
            Ok(())
        }
        None => Err(CError::new(
            TOKENIZERS_ERR_TOKENIZER,
            "with_truncation did not store the truncation params",
        )),
    }
}

/// Restores the saved truncation config when dropped (including on unwind). `restore` does it
/// explicitly on the success path so a failure there can be reported.
struct TruncationRestore<'a> {
    tokenizer: &'a mut Tokenizer,
    saved: Option<Option<TruncationParams>>,
}

impl<'a> TruncationRestore<'a> {
    fn restore(&mut self) -> CResult<()> {
        match self.saved.take() {
            Some(saved) => set_truncation(self.tokenizer, saved),
            None => Ok(()),
        }
    }
}

impl<'a> Drop for TruncationRestore<'a> {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// An empty id list is `{NULL, 0}`, so `token_ids == NULL` means "nothing to free".
fn into_result(ids: Vec<u32>) -> TokenizerEncodeResult {
    let len = ids.len();
    if len == 0 {
        return EMPTY_RESULT;
    }
    TokenizerEncodeResult {
        token_ids: Box::into_raw(ids.into_boxed_slice()) as *mut u32,
        len,
    }
}

const EMPTY_RESULT: TokenizerEncodeResult = TokenizerEncodeResult {
    token_ids: std::ptr::null_mut(),
    len: 0,
};

#[no_mangle]
extern "C" fn tokenizers_new_from_str(input_cstr: *const u8, len: usize) -> *mut TokenizerWrapper {
    guard_ptr(|| unsafe {
        let raw = bytes(input_cstr, len, "json")?;
        let json = String::from_utf8_lossy(raw);
        TokenizerWrapper::from_str(&json)
    })
}

#[no_mangle]
extern "C" fn byte_level_bpe_tokenizers_new_from_str(
    input_vocab_str: *const u8,
    len_vocab: usize,
    input_merges_str: *const u8,
    len_merges: usize,
    input_added_tokens_str: *const u8,
    len_added_tokens: usize,
) -> *mut TokenizerWrapper {
    guard_ptr(|| unsafe {
        let vocab = String::from_utf8_lossy(bytes(input_vocab_str, len_vocab, "vocab")?);
        let merges = String::from_utf8_lossy(bytes(input_merges_str, len_merges, "merges")?);
        let added_tokens = String::from_utf8_lossy(bytes(
            input_added_tokens_str,
            len_added_tokens,
            "added_tokens",
        )?);
        TokenizerWrapper::byte_level_bpe_from_str(&vocab, &merges, &added_tokens)
    })
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) -> i32 {
    guard_status(|| unsafe {
        let out = out_result.as_mut().ok_or_else(|| null_arg("result"))?;
        *out = EMPTY_RESULT;
        let wrapper = handle_mut(handle)?;
        let text = strict_str(input_cstr, len, "text")?;
        let ids = wrapper.encode(text, add_special_tokens != 0)?;
        *out = into_result(ids);
        Ok(())
    })
}

/// Shared body of `tokenizers_encode_batch` and `tokenizers_encode_batch_truncated`.
/// `max_length == 0` means no truncation: the plain encode_batch path, config untouched.
unsafe fn encode_batch_impl(
    handle: *mut TokenizerWrapper,
    input_cstr: *const *const u8,
    input_len: *const usize,
    num_seqs: usize,
    add_special_tokens: i32,
    max_length: usize,
    out_result: *mut TokenizerEncodeResult,
) -> CResult<()> {
    let wrapper = handle_mut(handle)?;
    if num_seqs == 0 {
        return Ok(());
    }
    if out_result.is_null() {
        return Err(null_arg("results"));
    }
    let outs = std::slice::from_raw_parts_mut(out_result, num_seqs);
    for o in outs.iter_mut() {
        *o = EMPTY_RESULT;
    }
    if input_cstr.is_null() {
        return Err(null_arg("data"));
    }
    if input_len.is_null() {
        return Err(null_arg("len"));
    }
    let ptrs = std::slice::from_raw_parts(input_cstr, num_seqs);
    let lens = std::slice::from_raw_parts(input_len, num_seqs);
    let mut texts: Vec<&str> = Vec::with_capacity(num_seqs);
    for (i, (&p, &n)) in ptrs.iter().zip(lens.iter()).enumerate() {
        texts.push(strict_str(p, n, &format!("data[{}]", i))?);
    }
    let encoded = if max_length == 0 {
        wrapper.encode_batch(texts, add_special_tokens != 0)?
    } else {
        wrapper.encode_batch_truncated(texts, add_special_tokens != 0, max_length)?
    };
    if encoded.len() != num_seqs {
        return Err(CError::new(
            TOKENIZERS_ERR_TOKENIZER,
            format!(
                "encode_batch returned {} results for {} inputs",
                encoded.len(),
                num_seqs
            ),
        ));
    }
    // Nothing below can fail, so no partial allocations are left on error.
    for (o, ids) in outs.iter_mut().zip(encoded.into_iter()) {
        *o = into_result(ids);
    }
    Ok(())
}

#[no_mangle]
extern "C" fn tokenizers_encode_batch(
    handle: *mut TokenizerWrapper,
    input_cstr: *const *const u8,
    input_len: *const usize,
    num_seqs: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) -> i32 {
    guard_status(|| unsafe {
        encode_batch_impl(
            handle,
            input_cstr,
            input_len,
            num_seqs,
            add_special_tokens,
            0,
            out_result,
        )
    })
}

#[no_mangle]
extern "C" fn tokenizers_encode_batch_truncated(
    handle: *mut TokenizerWrapper,
    input_cstr: *const *const u8,
    input_len: *const usize,
    num_seqs: usize,
    add_special_tokens: i32,
    max_length: usize,
    out_result: *mut TokenizerEncodeResult,
) -> i32 {
    guard_status(|| unsafe {
        encode_batch_impl(
            handle,
            input_cstr,
            input_len,
            num_seqs,
            add_special_tokens,
            max_length,
            out_result,
        )
    })
}

#[no_mangle]
extern "C" fn tokenizers_free_encode_results(results: *mut TokenizerEncodeResult, num_seqs: usize) {
    let _ = guard_status(|| unsafe {
        if results.is_null() || num_seqs == 0 {
            return Ok(());
        }
        let slice = std::slice::from_raw_parts_mut(results, num_seqs);
        for result in slice.iter_mut() {
            if !result.token_ids.is_null() {
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    result.token_ids,
                    result.len,
                )));
            }
            *result = EMPTY_RESULT;
        }
        Ok(())
    });
}

#[no_mangle]
extern "C" fn tokenizers_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: i32,
) -> i32 {
    guard_status(|| unsafe {
        let wrapper = handle_mut(handle)?;
        wrapper.decode_str.clear();
        let ids: &[u32] = if len == 0 {
            &[]
        } else if input_ids.is_null() {
            return Err(CError::new(
                TOKENIZERS_ERR_NULL_ARG,
                format!("ids is NULL with length {}", len),
            ));
        } else {
            std::slice::from_raw_parts(input_ids, len)
        };
        wrapper.decode(ids, skip_special_tokens != 0)
    })
}

#[no_mangle]
extern "C" fn tokenizers_get_decode_str(
    handle: *mut TokenizerWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard_status(|| unsafe {
        let wrapper = handle_mut(handle)?;
        if out_cstr.is_null() {
            return Err(null_arg("data"));
        }
        if out_len.is_null() {
            return Err(null_arg("len"));
        }
        *out_cstr = wrapper.decode_str.as_mut_ptr();
        *out_len = wrapper.decode_str.len();
        Ok(())
    })
}

#[no_mangle]
extern "C" fn tokenizers_free(wrapper: *mut TokenizerWrapper) {
    let _ = guard_status(|| unsafe {
        if !wrapper.is_null() {
            drop(Box::from_raw(wrapper));
        }
        Ok(())
    });
}

#[no_mangle]
extern "C" fn tokenizers_get_vocab_size(handle: *mut TokenizerWrapper, size: *mut usize) -> i32 {
    guard_status(|| unsafe {
        let wrapper = handle_mut(handle)?;
        let out = size.as_mut().ok_or_else(|| null_arg("size"))?;
        *out = wrapper.tokenizer.get_vocab_size(true);
        Ok(())
    })
}

#[no_mangle]
extern "C" fn tokenizers_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard_status(|| unsafe {
        let wrapper = handle_mut(handle)?;
        if out_cstr.is_null() {
            return Err(null_arg("data"));
        }
        if out_len.is_null() {
            return Err(null_arg("len"));
        }
        wrapper.id_to_token_result = wrapper.tokenizer.id_to_token(id).unwrap_or_default();
        *out_cstr = wrapper.id_to_token_result.as_mut_ptr();
        *out_len = wrapper.id_to_token_result.len();
        Ok(())
    })
}

/// Ids >= 2^31 do not fit the C API's `int32_t` (and 0xFFFFFFFF would read as -1, "not in
/// vocab"), so they are an error rather than a silent wrap.
fn id_to_i32(id: u32) -> CResult<i32> {
    i32::try_from(id).map_err(|_| {
        CError::new(
            TOKENIZERS_ERR_TOKENIZER,
            format!("token id {} does not fit in int32_t", id),
        )
    })
}

#[no_mangle]
extern "C" fn tokenizers_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const u8,
    len: usize,
    out_id: *mut i32,
) -> i32 {
    guard_status(|| unsafe {
        let wrapper = handle_mut(handle)?;
        let out = out_id.as_mut().ok_or_else(|| null_arg("id"))?;
        let token = String::from_utf8_lossy(bytes(token, len, "token")?);
        *out = match wrapper.tokenizer.token_to_id(&token) {
            Some(id) => id_to_i32(id)?,
            None => -1,
        };
        Ok(())
    })
}

#[no_mangle]
extern "C" fn tokenizers_get_last_error(out_cstr: *mut *const u8, out_len: *mut usize) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let (p, n) = LAST_ERROR
            .try_with(|e| match e.try_borrow() {
                Ok(s) => (s.as_ptr(), s.len()),
                Err(_) => (std::ptr::null(), 0),
            })
            .unwrap_or((std::ptr::null(), 0));
        if let Some(o) = out_cstr.as_mut() {
            *o = p;
        }
        if let Some(o) = out_len.as_mut() {
            *o = n;
        }
    }));
}

#[cfg(test)]
mod tests;

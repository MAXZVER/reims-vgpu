//! Resolve the translator's nil-texture ABI from the guest's argument table.
//! The image's size/pixels are not evidence of presence: a real 1x1 texture and
//! a nil binding served by a neutral image must remain distinguishable.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const PRESENCE_WORDS: usize = metal2vulkan::reflect::TEXTURE_ARGUMENT_COUNT_USIZE.div_ceil(32);

pub(super) struct TexturePresenceVariants {
    queries: Result<Vec<(usize, u32)>, &'static str>,
    words: Mutex<HashMap<[u32; PRESENCE_WORDS], Arc<Vec<u32>>>>,
}

impl TexturePresenceVariants {
    pub(super) fn new(words: &[u32]) -> Self {
        Self {
            queries: queries(words),
            words: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn resolve(
        &self,
        original: &Arc<Vec<u32>>,
        present: impl Fn(u32) -> bool,
    ) -> Result<Arc<Vec<u32>>, &'static str> {
        let queries = self.queries.as_ref().map_err(|e| *e)?;
        let mut nil = [0u32; PRESENCE_WORDS];
        for &(_, index) in queries {
            if !present(index) {
                nil[index as usize / 32] |= 1 << (index % 32);
            }
        }
        if nil == [0; PRESENCE_WORDS] {
            return Ok(original.clone());
        }
        let mut cache = self.words.lock().unwrap_or_else(|e| e.into_inner());
        Ok(cache
            .entry(nil)
            .or_insert_with(|| {
                let mut words = (**original).clone();
                for &(offset, index) in queries {
                    // OpSpecConstantTrue/False have the same three-word shape.
                    // Leave SpecId intact; each pipeline consumes this module's
                    // exact defaults, and its cache key includes these words.
                    let opcode = if nil[index as usize / 32] & (1 << (index % 32)) != 0 {
                        48
                    } else {
                        49
                    };
                    words[offset] = (3 << 16) | opcode;
                }
                Arc::new(words)
            })
            .clone())
    }
}

fn queries(words: &[u32]) -> Result<Vec<(usize, u32)>, &'static str> {
    const MALFORMED: &str = "texture_presence_malformed_spirv";
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(MALFORMED);
    }
    let mut ids = HashMap::new();
    let mut constants = HashMap::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let opcode = words[offset] & 0xffff;
        if count == 0 || count > words.len() - offset {
            return Err(MALFORMED);
        }
        let inst = &words[offset..offset + count];
        if opcode == 71 && count == 4 && inst[2] == 1 {
            if let Some(index) = metal2vulkan::reflect::texture_null_spec_index(inst[3]) {
                if ids.insert(inst[1], index).is_some() {
                    return Err("texture_presence_duplicate_spec_id");
                }
            }
        } else if opcode == 49 && count == 3 {
            constants.insert(inst[2], offset);
        }
        offset += count;
    }
    let mut result = Vec::new();
    for (id, index) in ids {
        let offset = constants
            .get(&id)
            .copied()
            .ok_or("texture_presence_spec_not_boolean_false")?;
        result.push((offset, index));
    }
    result.sort_unstable();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module() -> Arc<Vec<u32>> {
        Arc::new(vec![
            0x0723_0203,
            0x0001_0000,
            0,
            4,
            0,
            (4 << 16) | 71,
            2,
            1,
            metal2vulkan::reflect::texture_null_spec_id(3).unwrap(),
            (2 << 16) | 20,
            1,
            (3 << 16) | 49,
            1,
            2,
        ])
    }

    #[test]
    fn nil_is_distinct_from_a_real_one_pixel_texture_and_variants_are_reused() {
        let words = module();
        let variants = TexturePresenceVariants::new(&words);
        let present = variants.resolve(&words, |index| index == 3).unwrap();
        assert!(Arc::ptr_eq(&present, &words));
        let nil = variants.resolve(&words, |_| false).unwrap();
        assert_eq!(nil[11] & 0xffff, 48);
        assert_eq!(words[11] & 0xffff, 49, "the base module remains non-null");
        assert!(Arc::ptr_eq(
            &nil,
            &variants.resolve(&words, |_| false).unwrap()
        ));
        assert!(Arc::ptr_eq(
            &present,
            &variants.resolve(&words, |index| index == 3).unwrap()
        ));
    }

    #[test]
    fn unrelated_bindings_do_not_create_shader_variants() {
        let words = module();
        let variants = TexturePresenceVariants::new(&words);
        let a = variants
            .resolve(&words, |index| index == 3 || index == 8)
            .unwrap();
        let b = variants.resolve(&words, |index| index == 3).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn malformed_or_non_boolean_presence_constants_are_refused() {
        assert!(queries(&[0; 5]).is_err());
        let mut words = (*module()).clone();
        words[11] = (3 << 16) | 43;
        assert_eq!(
            queries(&words),
            Err("texture_presence_spec_not_boolean_false")
        );
    }
}

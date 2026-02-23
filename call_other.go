//go:build !linux || !amd64

package llama

import (
	"github.com/ebitengine/purego"
)

func registerInitFromModel(libHandle uintptr) error {
	purego.RegisterLibFunc(&llamaInitFromModelRaw, libHandle, "llama_init_from_model")
	return nil
}

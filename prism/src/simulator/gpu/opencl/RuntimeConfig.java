package simulator.gpu.opencl;

import java.util.EnumSet;

import parser.State;
import simulator.gpu.opencl.kernel.KernelException;
import simulator.gpu.opencl.kernel.PRNGType;

import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLDevice.Type;

public class RuntimeConfig
{
	static public final int DEFAULT_GLOBAL_WORK_SIZE = 100000;
	public int globalWorkSize = DEFAULT_GLOBAL_WORK_SIZE;
	static public final long DEFAULT_MAX_PATH_LENGTH = 1000;
	public long maxPathLength = DEFAULT_MAX_PATH_LENGTH;
	static public final int DEFAULT_SAMPLE_OFFSET = 0;
	public int sampleOffset = DEFAULT_SAMPLE_OFFSET;
	public State initialState = null;
	public Type deviceType = Type.CPU;
	public int registerCount = 0;
	public long localMemorySize = 0;
	public PRNGType prngType = null;
	public long prngSeed = 0;

	/**
	 * Direct method - known number of iterations.
	*/
	static public final int DEFAULT_DIRECT_CPU_GWSIZE = 100000;
	public int directMethodGWSizeCPU = DEFAULT_DIRECT_CPU_GWSIZE;
	static public final int DEFAULT_DIRECT_GPU_GWSIZE = 25000;
	public int directMethodGWSizeGPU = DEFAULT_DIRECT_GPU_GWSIZE;
	/**
	 * Number of iterations not known.
	 */
	static public final int DEFAULT_INDIRECT_CPU_GWSIZE = 25000;
	public int inDirectMethodGWSizeCPU = DEFAULT_INDIRECT_CPU_GWSIZE;
	static public final int DEFAULT_INDIRECT_GPU_GWSIZE = 25000;
	public int inDirectMethodGWSizeGPU = DEFAULT_INDIRECT_GPU_GWSIZE;
	static public final int DEFAULT_INDIRECT_RESULT_PERIOD = 1;
	public int inDirectResultCheckPeriod = DEFAULT_INDIRECT_RESULT_PERIOD;
	static public final int DEFAULT_INDIRECT_PATH_PERIOD = 3;
	public int inDirectPathCheckPeriod = DEFAULT_INDIRECT_PATH_PERIOD;

	public enum PRNG {
		MERSENNE_TWISTER, MWC64X
	}

	public RuntimeConfig()
	{
	}

	public RuntimeConfig(RuntimeConfig config)
	{
		this.globalWorkSize = config.globalWorkSize;
		this.maxPathLength = config.maxPathLength;
		this.initialState = config.initialState;
		this.deviceType = config.deviceType;
		this.registerCount = config.registerCount;
		this.localMemorySize = config.localMemorySize;
		this.prngType = config.prngType;
		this.prngSeed = config.prngSeed;
	}

	public void configDevice(CLDeviceWrapper dev) throws KernelException
	{
		CLDevice device = dev.getDevice();
		EnumSet<Type> types = device.getType();
		if (types.contains(Type.GPU)) {
			deviceType = Type.GPU;
			localMemorySize = device.getLocalMemSize();
		} else if (!types.contains(Type.CPU)) {
			throw new KernelException(String.format("Currently OpenCL Simulator does not support device %s which is not GPU or CPU", dev.getName()));
		}
	}
}
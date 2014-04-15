/**
 * 
 */
package simulator.gpu.opencl.kernel;

import java.util.ArrayList;
import java.util.List;

import org.bridj.Pointer;

import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.Include;
import simulator.gpu.opencl.kernel.expression.KernelComponent;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.CLVariable.Location;
import simulator.gpu.opencl.kernel.memory.PointerType;
import simulator.gpu.opencl.kernel.memory.RValue;
import simulator.gpu.opencl.kernel.memory.StdVariableType;
import simulator.gpu.opencl.kernel.memory.StdVariableType.StdType;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;

/**
 * @author mcopik
 *
 */
public class PRNGMersenneTwister extends PRNGType
{
	private static final List<Include> INCLUDES = new ArrayList<>();
	static {
		INCLUDES.add(new Include("MT_local.hcl", true));
	}
	private static final List<CLVariable> ADDITIONAL_ARGS = new ArrayList<>();
	static {
		//type
		PointerType ptr = new PointerType(new StdVariableType(StdType.UINT32));
		CLVariable seed = new CLVariable(ptr, "state");
		seed.setMemoryLocation(Location.GLOBAL);
		ADDITIONAL_ARGS.add(seed);
	}
	private CLBuffer<Integer> initBuffer = null;

	public PRNGMersenneTwister(String name)
	{
		super(name, INCLUDES, ADDITIONAL_ARGS);
	}

	public KernelComponent initializeGenerator()
	{
		return new Expression(String.format("MT_init( %s, %s );", varName, ADDITIONAL_ARGS.get(0).varName));
	}

	@Override
	public Expression deinitializeGenerator()
	{
		return new Expression(String.format("MT_save(%s);", varName));
	}

	@Override
	public List<Include> getIncludes()
	{
		List<Include> includes = new ArrayList<>();
		includes.add(new Include("MT_local.hcl", true));
		return includes;
	}

	@Override
	public int numbersPerRandomize()
	{
		return 1;
	}

	@Override
	public KernelComponent randomize() throws KernelException
	{
		return null;
	}

	@Override
	public CLValue getRandomInt(Expression randNumber, Expression max)
	{
		return new RValue(new Expression(String.format("MT_random_interval(%s,0,%d)", varName, max.getSource())));
	}

	@Override
	public CLValue getRandomFloat(Expression randNumber, Expression max)
	{
		return new RValue(new Expression(String.format("MT_rndFloat(%s)*%s", varName, max.getSource())));
	}

	@Override
	public CLValue getRandomUnifFloat(Expression randNumber)
	{
		return new RValue(new Expression(String.format("MT_rndFloat(%s)", varName)));
	}

	@Override
	public void setKernelArg(CLKernel kernel, int argNumber, int sampleOffset, int globalWorkSize, int localWorkSize)
	{
		if (initBuffer == null) {
			int[] initializeData = gpu.GPU.initializeMersenneTwister(globalWorkSize / localWorkSize, (int) random.randomUnifInt(Integer.MAX_VALUE));
			Pointer<Integer> ptr = Pointer.allocateInts(initializeData.length);
			ptr.setInts(initializeData);
			CLContext context = kernel.getProgram().getContext();
			initBuffer = context.createIntBuffer(CLMem.Usage.InputOutput, ptr, true);
			kernel.setArg(0, initBuffer);
		}
	}
}
package simulator.gpu.opencl.kernel;

import java.util.ArrayList;
import java.util.List;

import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.ExpressionList;
import simulator.gpu.opencl.kernel.expression.Include;
import simulator.gpu.opencl.kernel.expression.KernelComponent;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.RValue;
import simulator.gpu.opencl.kernel.memory.StdVariableType;
import simulator.gpu.opencl.kernel.memory.StdVariableType.StdType;

import com.nativelibs4java.opencl.CLKernel;

public class PRNGmwc64x extends PRNGType
{
	/**
	 * Because Java doesn't have unsigned int.
	 * Or unsigned long.
	 * Or just unsigned arithmetic.
	 * Why?
	 */
	public static final String RNG_MAX = "0xFFFFFFFF";
	long baseOffset;
	private static final List<Include> INCLUDES = new ArrayList<>();
	static {
		INCLUDES.add(new Include("mwc64x_rng.cl", true));
	}
	private static final List<CLVariable> ADDITIONAL_ARGS = new ArrayList<>();
	static {
		ADDITIONAL_ARGS.add(new CLVariable(new StdVariableType(StdType.UINT64), "seed"));
	}
	private static final String TYPE_NAME = "mwc64x_state_t";
	private long seed = -1;
	/**
	 * Equals to 2^40 - which gives interval for 2^23 samples.
	 */
	static public final long DEFAULT_RNG_OFFSET = 1099511627776L;
	private long rngOffset = DEFAULT_RNG_OFFSET;

	public PRNGmwc64x(String name)
	{
		super(name, INCLUDES, ADDITIONAL_ARGS);
	}

	public PRNGmwc64x(String name, long rngOffset)
	{
		super(name, INCLUDES, ADDITIONAL_ARGS);
		this.rngOffset = rngOffset;
	}

	public KernelComponent initializeGenerator()
	{
		ExpressionList list = new ExpressionList();
		list.addExpression(new Expression(String.format("%s %s;", TYPE_NAME, varName)));
		list.addExpression(new Expression(String.format("MWC64X_SeedStreams(&%s,%s,%sL);",
		//prng variable name
				varName,
				//uint64_t seed
				ADDITIONAL_ARGS.get(0).varName,
				//default range of samples for thread
				Long.toString(rngOffset))));
		return list;
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
		return new RValue(new Expression(String.format("floor(((float)MWC64X_NextUint(&%s))*%s/%s)", varName, max.getSource(), RNG_MAX)));
	}

	@Override
	public CLValue getRandomUnifFloat(Expression randNumber)
	{
		return new RValue(new Expression(String.format("((float)MWC64X_NextUint(&%s))/%s", varName, RNG_MAX)));
	}

	@Override
	public CLValue getRandomFloat(Expression randNumber, Expression max)
	{
		return new RValue(new Expression(String.format("((float)MWC64X_NextUint(&%s))*%s/%s", varName, max.getSource(), RNG_MAX)));
	}

	@Override
	public void setKernelArg(CLKernel kernel, int argNumber, int sampleOffset, int globalWorkSize, int localWorkSize)
	{
		if (seed == -1) {
			seed = (long) Math.floor(random.randomUnifDouble() * Long.MAX_VALUE);
		}
		kernel.setArg(argNumber, seed + sampleOffset);
	}
}